import cv2
import os
import time

class StorageManager:
    def __init__(self, data_dir="data"):
        """
        Feature 5: Dual-Stream Capture and Storage Sizing
        """
        self.data_dir = data_dir
        self.stream_a_dir = os.path.join(data_dir, "stream_a_continuous")
        self.stream_b_dir = os.path.join(data_dir, "stream_b_events")
        
        os.makedirs(self.stream_a_dir, exist_ok=True)
        os.makedirs(self.stream_b_dir, exist_ok=True)
        
    def save_stream_a(self, frame):
        """
        Stream A: Continuous, downscaled (480p), written to rolling buffer.
        In a real deployment, this pushes to an H.264 video writer.
        For prototype, we just update a single file to prevent disk fill.
        """
        downscaled = cv2.resize(frame, (640, 480))
        cv2.imwrite(os.path.join(self.stream_a_dir, "rolling_buffer.jpg"), downscaled)
        
    def save_event_frame(self, frame, event_tag):
        """
        Stream B: Full native resolution, single frames on AI flag.
        """
        timestamp = int(time.time())
        filename = f"{event_tag}_{timestamp}.jpg"
        filepath = os.path.join(self.stream_b_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"[Storage] Saved high-res event frame: {filepath}")
