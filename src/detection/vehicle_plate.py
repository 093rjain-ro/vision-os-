import cv2
import os
import numpy as np
import easyocr
from ultralytics import YOLO

class VehiclePlateRecognizer:
    def __init__(self, model_path="license_plate_detector.pt"):
        """
        Feature 6: Tier 1 Plate Detection (YOLO/Cascade) + Tier 2 EasyOCR.
        Replaces full-vehicle crop guessing with a specific plate detector.
        """
        self.reader = easyocr.Reader(['en'], gpu=False) # Tier 2 OCR
        
        # Load plate-specific YOLO model if available
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.is_yolo = True
            except:
                self.is_yolo = False
        else:
            self.is_yolo = False
            
        # Fallback to Haar Cascade if YOLO plate model is missing
        if not self.is_yolo:
            cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect_plates(self, frame):
        """
        Returns bounding boxes of license plates directly.
        """
        boxes = []
        if self.is_yolo:
            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                if float(box.conf[0]) > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in plates:
                boxes.append((x, y, x + w, y + h))
                
        return boxes

    def read_plate(self, plate_crop):
        """
        Tier 2: EasyOCR on the specific plate crop.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0
            
        results = self.reader.readtext(plate_crop)
        if not results:
            return None, 0.0
            
        # Sort by confidence and get highest
        best_result = max(results, key=lambda x: x[2])
        text, conf = best_result[1], best_result[2]
        
        # Clean text (remove spaces/special chars)
        clean_text = "".join(e for e in text if e.isalnum()).upper()
        return clean_text, float(conf)
