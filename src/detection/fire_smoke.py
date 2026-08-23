import cv2
import numpy as np

class FireSmokeDetector:
    def __init__(self):
        """
        Feature 2: Fire & Smoke Detection
        In production, this loads a custom-labeled YOLOv8n-fire model.
        For this prototype, we simulate a lightweight CNN passing over frames.
        """
        self.frame_count = 0
        self.consecutive_detections = 0
        self.CONFIRM_THRESHOLD = 3 # Needs 3 consecutive frames to alert

    def detect(self, frame):
        """
        Runs on every Nth frame (simulated).
        """
        self.frame_count += 1
        
        # Skip frames to save compute (e.g., every 5th frame)
        if self.frame_count % 5 != 0:
            return False, 0.0

        # Simulated detection logic
        # For demo purposes, we will trigger fire if image has a massive amount of bright red/orange
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 150, 150])
        upper_fire = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
        
        detected = ratio > 0.05 # 5% of screen is fire-colored
        confidence = min(ratio * 10, 1.0)

        if detected:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        # Two-stage confirmation (Feature 2)
        if self.consecutive_detections >= self.CONFIRM_THRESHOLD:
            return True, confidence
            
        return False, 0.0
