import sqlite3
import datetime
import os

class DatabaseLogger:
    def __init__(self, db_path="data/vision_os.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # General Events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                identifier TEXT,
                confidence REAL,
                action_taken TEXT
            )
        ''')

        # Smart Attendance (Feature 1)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                person_id TEXT,
                camera_id TEXT,
                embedding BLOB
            )
        ''')
        
        # Alert Queue for Cellular Store-and-Forward (Feature 7)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_event(self, event_type, identifier, confidence, action_taken):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO events (timestamp, event_type, identifier, confidence, action_taken) VALUES (?, ?, ?, ?, ?)', 
                       (timestamp, event_type, identifier, confidence, action_taken))
        conn.commit()
        conn.close()
        print(f"[DB] Logged {event_type}: {identifier}")

    def log_attendance(self, person_id, camera_id, embedding_bytes):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO attendance (timestamp, person_id, camera_id, embedding) VALUES (?, ?, ?, ?)', 
                       (timestamp, person_id, camera_id, embedding_bytes))
        conn.commit()
        conn.close()
        print(f"[DB] Attendance Logged: {person_id}")

    def queue_alert(self, payload_json):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO alert_queue (payload, status) VALUES (?, ?)', (payload_json, 'pending'))
        conn.commit()
        conn.close()
