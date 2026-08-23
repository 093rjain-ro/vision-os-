import cv2
import easyocr
from ultralytics import YOLO

class VehiclePlateRecognizer:
    def __init__(self, model_path="yolov8n.pt", ocr_langs=['en']):
        """
        model_path: Path to YOLOv8 weights. In a real scenario, this would be a model trained to detect license plates.
        For this prototype, we use standard YOLOv8n to detect vehicles, and then assume plates are in a specific region,
        or ideally use a custom model that detects 'license_plate' class.
        """
        try:
            self.model = YOLO(model_path)
            # COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
            self.vehicle_classes = [2, 3, 5, 7] 
            print("YOLO model loaded.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

        print("Loading EasyOCR. This may take a moment...")
        self.reader = easyocr.Reader(ocr_langs, gpu=False) # Set gpu=True if CUDA available
        
    def detect_vehicles(self, frame):
        """Returns bounding boxes of detected vehicles."""
        if self.model is None:
            return []
            
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in self.vehicle_classes and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes
        
    def read_plate(self, plate_crop):
        """
        Reads text from a cropped image of a license plate.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None
            
        # Optional: Convert to grayscale for OCR
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        
        # Extract text with highest confidence
        best_text = None
        best_conf = 0.0
        
        for (bbox, text, prob) in results:
            if prob > best_conf:
                best_conf = prob
                best_text = text
                
        # Clean up text (remove spaces, non-alphanumeric)
        if best_text:
            best_text = ''.join(e for e in best_text if e.isalnum()).upper()
            
        return best_text, best_conf
