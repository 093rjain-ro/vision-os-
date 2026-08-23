import cv2
import time

class VideoStream:
    def __init__(self, source=0, resolution=(640, 480)):
        self.source = source
        self.resolution = resolution
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Warning: Could not open video source {self.source}")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
    def get_frame(self):
        """Reads a frame from the video source."""
        if not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        """Releases the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()
