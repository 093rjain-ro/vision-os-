import numpy as np
from src.core.tracker import CentroidTracker
from src.detection.face_attendance import SmartAttendance
import os

def test_centroid_tracker():
    tracker = CentroidTracker(max_disappeared=2)
    
    # Frame 1: One object
    rects1 = [(10, 10, 20, 20)]
    assigned, evicted = tracker.update(rects1)
    assert len(assigned) == 1
    assert assigned[0][0] == 0 # ID should be 0
    assert len(evicted) == 0
    
    # Frame 2: Same object moved slightly
    rects2 = [(12, 12, 22, 22)]
    assigned, evicted = tracker.update(rects2)
    assert len(assigned) == 1
    assert assigned[0][0] == 0 # ID should still be 0
    
    # Frame 3: Object disappears
    assigned, evicted = tracker.update([])
    assert len(assigned) == 0
    assert len(evicted) == 0 # Not evicted yet, only 1 frame disappeared
    
    # Frame 4: Still disappeared
    assigned, evicted = tracker.update([])
    
    # Frame 5: Still disappeared -> triggers eviction since max_disappeared is 2
    assigned, evicted = tracker.update([])
    assert 0 in evicted

def test_smart_attendance():
    # Make sure enrollment dir exists
    os.makedirs("data/enrolled_faces", exist_ok=True)
    
    att = SmartAttendance("config/config.yaml")
    
    # Clean up mock if exists
    mock_file = "data/enrolled_faces/TEST_001.npy"
    if os.path.exists(mock_file):
        os.remove(mock_file)
        
    # Inject a known embedding directly instead of enrolling via image
    emb1 = np.ones(512)
    att.enrolled_faces["TEST_001"] = emb1
    att.match_threshold = 0.6
    
    # Exact match
    match_id, conf = att.match_face(emb1)
    assert match_id == "TEST_001", f"Expected TEST_001, got {match_id}"
    assert conf > 0.99
    
    # Opposite match
    emb2 = -np.ones(512)
    match_id, conf = att.match_face(emb2)
    assert match_id == "UNKNOWN"
    assert conf < 0.6

if __name__ == "__main__":
    test_centroid_tracker()
    test_smart_attendance()
    print("All smoke tests passed!")
