import numpy as np
from src.core.tracker import CentroidTracker
from src.detection.face_attendance import SmartAttendance
from src.db.logger import DatabaseLogger
import os
import re

def test_centroid_tracker():
    tracker = CentroidTracker(max_disappeared=2)
    rects1 = [(10, 10, 20, 20)]
    assigned, evicted = tracker.update(rects1)
    assert len(assigned) == 1
    assert assigned[0][0] == 0
    assert len(evicted) == 0
    
    rects2 = [(12, 12, 22, 22)]
    assigned, evicted = tracker.update(rects2)
    assert len(assigned) == 1
    assert assigned[0][0] == 0
    
    assigned, evicted = tracker.update([])
    assert len(assigned) == 0
    assert len(evicted) == 0
    
    assigned, evicted = tracker.update([])
    assigned, evicted = tracker.update([])
    assert 0 in evicted

def test_smart_attendance_and_visitors():
    os.makedirs("data/enrolled_faces", exist_ok=True)
    logger = DatabaseLogger("data/test_vision_os.db")
    
    # Cleanup DB
    conn = logger.conn if hasattr(logger, 'conn') else None
    if not conn:
        import sqlite3
        conn = sqlite3.connect(logger.db_path)
    conn.execute("DELETE FROM visitors")
    conn.commit()
    conn.close()
    
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
    
    # Cleanup test db
    os.remove("data/test_vision_os.db")

def test_alpr_regex():
    import yaml
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    plate_regex = cfg['alpr']['plate_format_regex']
    
    # Valid Indian plate
    assert re.match(plate_regex, "MH12AB1234")
    assert re.match(plate_regex, "KA01X9999")
    
    # Invalid plates
    assert not re.match(plate_regex, "INVALID")
    assert not re.match(plate_regex, "M12AB1234")
    assert not re.match(plate_regex, "MH12AB123")

if __name__ == "__main__":
    test_centroid_tracker()
    test_smart_attendance_and_visitors()
    test_alpr_regex()
    print("All extended smoke tests passed!")
