import cv2
import numpy as np

from ultralytics import YOLO

class SmartAttendance:
    def __init__(self):
        """
        Feature 1: Smart Attendance
        Uses YOLOv8 to find persons, and assumes the top 15% of the bounding box is the head.
        """
        try:
            self.model = YOLO("yolov8n.pt")
            self.person_class = 0
        except:
            self.model = None
            
        # Dummy authorized database
        self.authorized_faces = {
            "EMP_001": np.random.rand(128),
            "EMP_002": np.random.rand(128)
        }

    def detect_faces(self, frame):
        """Returns approximate head bounding boxes (x, y, w, h)"""
        if self.model is None: return []
        
        results = self.model(frame, verbose=False)[0]
        faces = []
        for box in results.boxes:
            if int(box.cls[0]) == self.person_class and float(box.conf[0]) > 0.3: # Lowered to 0.3
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                # Approximate face as top 15% of person bounding box
                face_h = int(h * 0.15)
                # Keep y1 same, adjust x to be centered
                face_w = int(w * 0.5)
                face_x = x1 + int((w - face_w) / 2)
                faces.append((face_x, y1, face_w, face_h))
        return faces

    def generate_embedding(self, face_crop):
        """
        Mocks a MobileFaceNet 128-d embedding generation.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        return np.random.rand(128) # Simulated embedding

    def match_face(self, embedding):
        """
        Locally matches vector against encrypted DB (simulated).
        """
        if embedding is None: return None, 0
        
        # Simulated match logic - random chance for prototype demo
        if np.random.random() > 0.8:
            return "EMP_001", 0.95
        return "UNKNOWN", 0.0
