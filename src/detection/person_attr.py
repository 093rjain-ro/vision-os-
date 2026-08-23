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

    def extract_dominant_color(self, person_crop):
        """
        Estimates the dominant clothing color of a person crop.
        Used for safety compliance (e.g. checking for hi-vis colors).
        """
        if person_crop is None or person_crop.size == 0:
            return "unknown"
            
        # Focus on the upper half for shirt/jacket color
        h, w = person_crop.shape[:2]
        upper_half = person_crop[0:max(1, h//2), :]
        
        hsv = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)
        
        best_color = "unknown"
        max_pixels = 0
        
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            if len(ranges) == 4: # Red has two ranges in HSV
                mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges[0], ranges[1])
                
            pixels = cv2.countNonZero(mask)
            if pixels > max_pixels:
                max_pixels = pixels
                best_color = color_name
                
        # Threshold to ensure it's a significant portion
        if max_pixels > (h//2 * w) * 0.1: # At least 10% of upper half
            return best_color
        return "unknown"
