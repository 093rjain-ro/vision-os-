import cv2
import yaml
import time
import queue
import threading
from core.video_stream import VideoStream
from core.enhancer import ImageEnhancer
from core.tracker import CentroidTracker
from core.storage_manager import StorageManager
from detection.vehicle_plate import VehiclePlateRecognizer
from detection.person_attr import PersonAttributeExtractor
from detection.face_attendance import SmartAttendance
from detection.fire_smoke import FireSmokeDetector
from hardware.actuator import Actuator
from hardware.alert import AlertNotifier
from hardware.sensors import HardwareSensors
from db.logger import DatabaseLogger

class VisionOSPipeline:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.stream = VideoStream(source=self.config['camera']['source'])
        self.enhancer = ImageEnhancer()
        self.tracker = CentroidTracker()
        self.vehicle_tracker = CentroidTracker(max_disappeared=10)
        self.vehicle_read_counts = {}
        self.vehicle_best_reads = {}
        self.storage = StorageManager()
        
        # Detection Models
        self.plate_recognizer = VehiclePlateRecognizer()
        self.person_extractor = PersonAttributeExtractor()
        self.attendance = SmartAttendance()
        self.fire_detector = FireSmokeDetector()
        
        # Hardware & Logging
        self.logger = DatabaseLogger()
        self.actuator = Actuator(config_path)
        self.notifier = AlertNotifier(config_path=config_path)
        self.sensors = HardwareSensors(self.actuator, self.notifier, self.logger)
        
        self.auth_plates = self.config['access_control'].get('authorized_plates', [])
        
        # Cooldown state dictionaries to prevent DB spam
        self.cooldown_cfg = self.config.get('cooldowns', {'attendance_sec': 300, 'alert_sec': 60})
        self.last_event_times = {}
        self.alerted_persons = set()
        
        # Purge old visitors (Feature 9)
        retention = self.config.get('visitor_management', {}).get('visitor_retention_days', 30)
        self.logger.purge_old_visitors(retention)
        
        # Multithreading queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.write_queue = queue.Queue(maxsize=10)
        self.running = False

    def check_cooldown(self, dedup_key, cooldown_sec):
        now = time.time()
        last_seen = self.last_event_times.get(dedup_key, 0)
        if now - last_seen > cooldown_sec:
            self.last_event_times[dedup_key] = now
            return True
        return False
        
    def capture_loop(self):
        while self.running:
            ret, frame = self.stream.get_frame()
            if not ret: 
                break
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass # Drop oldest or just skip frame
                
    def write_loop(self):
        while self.running or not self.write_queue.empty():
            try:
                frame, save_event = self.write_queue.get(timeout=1.0)
                self.storage.save_stream_a(frame)
                cv2.imwrite("data/latest_frame.jpg", frame)
                if save_event:
                    self.storage.save_event_frame(frame, "EVENT")
            except queue.Empty:
                continue

    def run(self):
        print("Starting Advanced Vision OS Pipeline (Multithreaded)...")
        self.running = True
        
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        write_thread = threading.Thread(target=self.write_loop, daemon=True)
        
        capture_thread.start()
        write_thread.start()
        
        try:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    if not capture_thread.is_alive():
                        break
                    continue
                    
                event_triggered = False

                # Feature 2: Fire & Smoke
                is_fire, fire_conf = self.fire_detector.detect(frame)
                if is_fire:
                    cv2.putText(frame, "FIRE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if self.check_cooldown("fire_alert", self.cooldown_cfg.get('alert_sec', 60)):
                        self.actuator.trigger_alert()
                        self.notifier.send_alert("FIRE/SMOKE DETECTED", "Fire Alert", self.logger)
                        self.logger.log_event("Fire Alert", "Fire/Smoke", fire_conf, "Alarm Latched")
                        event_triggered = True
                    
                # Feature 1: Smart Attendance
                faces = self.attendance.detect_faces(frame)
                for face_obj in faces:
                    x, y, x2, y2 = face_obj.bbox.astype(int)
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 2)
                    
                    emb = self.attendance.generate_embedding(face_obj)
                    person_id, conf, person_type = self.attendance.match_face(emb, self.logger)
                    
                    if person_id != "UNKNOWN":
                        # Determine display label
                        label = person_id
                        if person_type == "VISITOR":
                            # Check if regular visitor
                            _, counts = self.logger.get_all_visitors()
                            count = counts.get(person_id, 1)
                            if count >= self.config.get('visitor_management', {}).get('regular_visitor_threshold', 10):
                                label = f"Regular: {person_id}"
                            else:
                                label = f"Visitor: {person_id}"
                        
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Logging & Alert logic
                        dedup_key = f"attendance_{person_id}"
                        if self.check_cooldown(dedup_key, self.cooldown_cfg.get('attendance_sec', 300)):
                            if person_type == "EMPLOYEE":
                                self.logger.log_attendance(person_id, self.config['system']['camera_id'], b"mock_blob")
                            elif person_type == "NEW":
                                self.notifier.send_alert(f"New/unrecognized person detected: {person_id}", "New Visitor", self.logger)
                                self.logger.log_event("New Visitor", person_id, 1.0, "Alert Sent")
                                event_triggered = True

                import re
                plate_boxes = self.plate_recognizer.detect_plates(frame)
                assigned_plates, evicted_plates = self.vehicle_tracker.update(plate_boxes)
                
                for ev_id in evicted_plates:
                    self.vehicle_read_counts.pop(ev_id, None)
                    self.vehicle_best_reads.pop(ev_id, None)
                    
                alpr_cfg = self.config.get('alpr', {})
                ocr_freq = alpr_cfg.get('ocr_every_n_frames', 5)
                min_conf = alpr_cfg.get('min_plate_confidence', 0.5)
                plate_regex = alpr_cfg.get('plate_format_regex', '^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')
                
                for obj_id, (x1, y1, x2, y2) in assigned_plates:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    count = self.vehicle_read_counts.get(obj_id, 0)
                    
                    if count % ocr_freq == 0:
                        plate_crop = frame[y1:y2, x1:x2]
                        enhanced_crop = self.enhancer.enhance(plate_crop) # Enhancer ONLY called when doing OCR
                        plate_text, conf = self.plate_recognizer.read_plate(enhanced_crop)
                        
                        if plate_text and conf >= min_conf:
                            if re.match(plate_regex, plate_text):
                                self.vehicle_best_reads[obj_id] = (plate_text, conf)
                                
                                dedup_key = f"plate_{obj_id}"
                                if self.check_cooldown(dedup_key, self.cooldown_cfg.get('alert_sec', 60)):
                                    if plate_text in self.auth_plates:
                                        self.actuator.open_barrier()
                                        self.logger.log_event("Access Granted", plate_text, conf, "Barrier Opened")
                                    else:
                                        self.logger.log_event("Access Denied", plate_text, conf, "None")
                                    event_triggered = True
                            
                    self.vehicle_read_counts[obj_id] = count + 1
                    
                    best_read = self.vehicle_best_reads.get(obj_id)
                    if best_read:
                        cv2.putText(frame, f"{best_read[0]} ({best_read[1]:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                # Intruder detection
                person_boxes = self.person_extractor.detect_persons(frame)
                
                # New CentroidTracker return format
                assigned_objects, evicted_ids = self.tracker.update(person_boxes)
                
                # Remove evicted IDs from alerted set (Fix 3)
                self.alerted_persons.difference_update(evicted_ids)
                
                for obj_id, (x1, y1, x2, y2) in assigned_objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    person_crop = frame[y1:y2, x1:x2]
                    
                    color = self.person_extractor.extract_dominant_color(person_crop)
                    cv2.putText(frame, f"ID {obj_id}: {color}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
                    unauth = self.config['access_control'].get('unauthorized_attributes', [])
                    for attr in unauth:
                        if attr.get('color') == color:
                            if obj_id not in self.alerted_persons:
                                self.alerted_persons.add(obj_id)
                                self.actuator.trigger_alert()
                                self.notifier.send_alert(f"Unauthorized person: {color}", "Security Alert", self.logger)
                                self.logger.log_event("Security Alert", f"Person ID {obj_id}: {color}", 1.0, "Alert Triggered")
                                event_triggered = True
                            break

                # Send frame to writer thread
                try:
                    self.write_queue.put((frame, event_triggered), block=False)
                except queue.Full:
                    pass
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.cleanup()
            
    def cleanup(self):
        self.stream.release()
        self.sensors.stop()
        if hasattr(self, 'notifier'):
            self.notifier.close()
