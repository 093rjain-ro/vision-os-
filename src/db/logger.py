import sqlite3
import os
from datetime import datetime, timedelta
import numpy as np

class DatabaseLogger:
    def __init__(self, db_path="data/vision_os.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      event_type TEXT,
                      identifier TEXT,
                      confidence REAL,
                      action_taken TEXT)''')
                      
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      person_id TEXT,
                      camera_id TEXT,
                      face_embedding BLOB)''')

        c.execute('''CREATE TABLE IF NOT EXISTS alert_queue
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      payload TEXT,
                      status TEXT)''')

        c.execute('''CREATE TABLE IF NOT EXISTS visitors
                     (visitor_id TEXT PRIMARY KEY,
                      embedding BLOB,
                      first_seen TEXT,
                      last_seen TEXT,
                      seen_count INTEGER)''')

        conn.commit()
        conn.close()

    def get_all_visitors(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT visitor_id, embedding, seen_count FROM visitors")
        rows = c.fetchall()
        conn.close()
        
        visitors = {}
        counts = {}
        for row in rows:
            vid = row[0]
            emb = np.frombuffer(row[1], dtype=np.float32)
            visitors[vid] = emb
            counts[vid] = row[2]
        return visitors, counts

    def update_visitor(self, visitor_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("UPDATE visitors SET last_seen=?, seen_count=seen_count+1 WHERE visitor_id=?", (now, visitor_id))
        conn.commit()
        conn.close()

    def add_new_visitor(self, embedding):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        c.execute("SELECT COUNT(*) FROM visitors")
        count = c.fetchone()[0]
        vid = f"VISITOR_{count+1:05d}"
        
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        c.execute("INSERT INTO visitors (visitor_id, embedding, first_seen, last_seen, seen_count) VALUES (?, ?, ?, ?, ?)",
                  (vid, emb_bytes, now, now, 1))
        conn.commit()
        conn.close()
        return vid

    def purge_old_visitors(self, retention_days):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("DELETE FROM visitors WHERE last_seen < ?", (cutoff,))
        conn.commit()
        conn.close()

    def log_event(self, event_type, identifier, confidence, action):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO events (timestamp, event_type, identifier, confidence, action_taken) VALUES (?, ?, ?, ?, ?)",
                  (timestamp, event_type, identifier, confidence, action))
        conn.commit()
        conn.close()
        
    def log_attendance(self, person_id, camera_id, face_embedding):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO attendance (timestamp, person_id, camera_id, face_embedding) VALUES (?, ?, ?, ?)",
                  (timestamp, person_id, camera_id, face_embedding))
        conn.commit()
        conn.close()
        
    def queue_alert(self, payload_json):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO alert_queue (payload, status) VALUES (?, ?)', (payload_json, 'pending'))
        conn.commit()
        conn.close()
