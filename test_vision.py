import numpy as np
from src.core.tracker import CentroidTracker
from src.detection.face_attendance import SmartAttendance
from src.detection.person_attr import PersonAttributeExtractor
from src.db.logger import DatabaseLogger
import os
import re
import cv2
import yaml

def test_centroid_tracker():
    tracker = CentroidTracker(max_disappeared=2)
    rects1 = [(10, 10, 20, 20)]
    assigned, evicted = tracker.update(rects1)
    assert len(assigned) == 1
    assert assigned[0][0] == 0
    assert len(evicted) == 0
    
    # Object moves slightly
    rects2 = [(12, 12, 22, 22)]
    assigned, evicted = tracker.update(rects2)
    assert len(assigned) == 1
    assert assigned[0][0] == 0
    
    # Object disappears
    assigned, evicted = tracker.update([])
    
    # Object formally evicted
    assigned, evicted = tracker.update([])
    assigned, evicted = tracker.update([])
    assert 0 in evicted

def test_smart_attendance_and_visitors():
    os.makedirs("data/enrolled_faces", exist_ok=True)
    logger = DatabaseLogger("data/test_vision_os.db")
    
    att = SmartAttendance("config/config.yaml")
    
    # Mock Enrollment
    emb1 = np.ones(512)
    att.enrolled_faces["TEST_001"] = emb1
    att.match_threshold = 0.6
    
    # Test Employee Match
    match_id, conf, p_type = att.match_face(emb1, logger)
    assert match_id == "TEST_001"
    assert p_type == "EMPLOYEE"
    
    # Test New Visitor
    emb2 = -np.ones(512)
    match_id, conf, p_type = att.match_face(emb2, logger)
    assert p_type == "NEW"
    assert match_id.startswith("VISITOR_")
    
    # Test Recurring Visitor
    match_id2, conf2, p_type2 = att.match_face(emb2, logger)
    assert p_type2 == "VISITOR"
    assert match_id2 == match_id
    
    if os.path.exists("data/test_vision_os.db"):
        os.remove("data/test_vision_os.db")

def test_alpr_regex():
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    plate_regex = cfg['alpr']['plate_format_regex']
    
    # Valid Indian plate
    assert bool(re.match(plate_regex, "MH12AB1234"))
    assert bool(re.match(plate_regex, "KA01X9999"))
    
    # Invalid plates
    assert not bool(re.match(plate_regex, "INVALID"))
    assert not bool(re.match(plate_regex, "M12AB1234"))
    assert not bool(re.match(plate_regex, "MH12AB123"))

def test_person_attribute_extractor():
    extractor = PersonAttributeExtractor(model_path="dummy.pt") # Dummy model so it won't crash
    
    # Create a dummy RED image crop
    dummy_crop_red = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_crop_red[:] = (0, 0, 255) # BGR
    
    color = extractor.extract_dominant_color(dummy_crop_red)
    assert color == "red"
    
    # Create a dummy BLUE image crop
    dummy_crop_blue = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_crop_blue[:] = (255, 0, 0) # BGR
    
    color_blue = extractor.extract_dominant_color(dummy_crop_blue)
    assert color_blue == "blue"

def test_database_logger():
    db_path = "data/test_logger.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    logger = DatabaseLogger(db_path)
    logger.log_event("Security Alert", "Person ID 5: red", 1.0, "Alert Triggered")
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT event_type, identifier FROM events")
    row = cursor.fetchone()
    assert row[0] == "Security Alert"
    assert row[1] == "Person ID 5: red"
    
    conn.close()
    os.remove(db_path)

if __name__ == "__main__":
    print("Running extended test suite...")
    test_centroid_tracker()
    print("✓ Centroid Tracker passed")
    
    test_smart_attendance_and_visitors()
    print("✓ Smart Attendance passed")
    
    test_alpr_regex()
    print("✓ ALPR Regex passed")
    
    test_person_attribute_extractor()
    print("✓ Person Attribute Extractor passed")
    
    test_database_logger()
    print("✓ Database Logger passed")
    
    print("\nAll tests passed successfully! 🚀")
