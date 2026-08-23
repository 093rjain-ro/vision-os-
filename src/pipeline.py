import cv2
import yaml
import time
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
        self.storage = StorageManager()
        
        # Detection Models (Tier 1 & 2 Hybrid - Feature 6)
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
        self.alerted_persons = set() # Trackers
        
    def check_cooldown(self, dedup_key, cooldown_sec):
        now = time.time()
        last_seen = self.last_event_times.get(dedup_key, 0)
        if now - last_seen > cooldown_sec:
            self.last_event_times[dedup_key] = now
            return True
        return False
        
    def run(self):
        print("Starting Advanced Vision OS Pipeline with Cooldowns...")
        
        while True:
            ret, frame = self.stream.get_frame()
            if not ret: break
                
            # Feature 5: Dual-Stream (Stream A continuous save)
            self.storage.save_stream_a(frame)
            event_triggered = False

            # Feature 2: Fire & Smoke (Tier 1)
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
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                crop = frame[y:y+h, x:x+w]
                emb = self.attendance.generate_embedding(crop)
                person_id, conf = self.attendance.match_face(emb)
                
                if person_id != "UNKNOWN":
                    cv2.putText(frame, f"Logged: {person_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    dedup_key = f"attendance_{person_id}"
                    if self.check_cooldown(dedup_key, self.cooldown_cfg.get('attendance_sec', 300)):
                        self.logger.log_attendance(person_id, self.config['system']['camera_id'], b"mock_blob")
                        event_triggered = True

            # Detect vehicles
            vehicle_boxes = self.plate_recognizer.detect_vehicles(frame)
            for (x1, y1, x2, y2) in vehicle_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                h_b = y2 - y1
                plate_crop = frame[y1 + h_b//2 : y2, x1:x2]
                enhanced_crop = self.enhancer.enhance(plate_crop)
                plate_text, conf = self.plate_recognizer.read_plate(enhanced_crop)
                if plate_text:
                    cv2.putText(frame, f"{plate_text} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if plate_text in self.auth_plates:
                        self.actuator.open_barrier()
                        self.logger.log_event("Access Granted", plate_text, conf, "Barrier Opened")
                    else:
                        self.logger.log_event("Access Denied", plate_text, conf, "None")
                    event_triggered = True
                        
            # Detect persons (Intruder detection)
            person_boxes = self.person_extractor.detect_persons(frame)
            tracked_objects = self.tracker.update(person_boxes)
            
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Match box to tracker ID
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                person_id = None
                for obj_id, centroid in tracked_objects.items():
                    if centroid[0] == cX and centroid[1] == cY:
                        person_id = obj_id
                        break
                
                person_crop = frame[y1:y2, x1:x2]
                gender = self.person_extractor.estimate_gender(person_crop)
                
                cv2.putText(frame, f"ID {person_id}: {gender}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                unauth = self.config['access_control'].get('unauthorized_attributes', [])
                for attr in unauth:
                    if attr.get('gender') == gender:
                        if person_id not in getattr(self, 'alerted_persons', set()):
                            if not hasattr(self, 'alerted_persons'): self.alerted_persons = set()
                            self.alerted_persons.add(person_id)
                            
                            self.actuator.trigger_alert()
                            self.notifier.send_alert(f"Unauthorized person: {gender}", "Security Alert", self.logger)
                            self.logger.log_event("Security Alert", f"Person ID {person_id}: {gender}", 1.0, "Alert Triggered")
                            event_triggered = True
                        break

            # Feature 5: Dual-Stream (Stream B high-res save on event)
            if event_triggered:
                self.storage.save_event_frame(frame, "EVENT")

            cv2.imwrite("data/latest_frame.jpg", frame)
                
        self.cleanup()
        
    def cleanup(self):
        self.stream.release()
        self.sensors.stop()
