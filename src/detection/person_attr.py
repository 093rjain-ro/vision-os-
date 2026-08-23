import cv2
import numpy as np
from ultralytics import YOLO

class PersonAttributeExtractor:
    def __init__(self, model_path="yolov8n.pt"):
        try:
            self.model = YOLO(model_path)
            self.person_class = 0 # COCO class 0 is person
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

        # Basic color ranges in HSV
        self.color_ranges = {
            "red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (179, 255, 255)],
            "blue": [(100, 150, 0), (140, 255, 255)],
            "green": [(35, 100, 100), (85, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 30)],
            "white": [(0, 0, 200), (180, 30, 255)]
        }

    def detect_persons(self, frame):
        """Returns bounding boxes of detected persons."""
        if self.model is None:
            return []
            
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == self.person_class and conf > 0.3: # Lowered to 0.3 for difficult poses
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes

    def estimate_gender(self, person_crop):
        """
        Estimates gender. In a production build, this would use a dedicated CNN model 
        (e.g., FairFace or DeepFace) to analyze the face crop. 
        For this prototype, without downloading a heavy model, we mock it based on image brightness 
        just to demonstrate the routing logic on the dashboard.
        """
        if person_crop is None or person_crop.size == 0:
            return "Men"
            
        # Dummy classification for prototype demonstration
        avg_brightness = np.mean(person_crop)
        if avg_brightness > 120:
            return "Men"
        else:
            return "Women"
