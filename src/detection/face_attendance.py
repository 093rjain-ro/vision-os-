import cv2
import numpy as np
import os
import yaml
from scipy.spatial.distance import cosine
import insightface
from insightface.app import FaceAnalysis

class SmartAttendance:
    def __init__(self, config_path="config/config.yaml"):
        """
        Feature 1: Smart Attendance with Real InsightFace Embeddings
        """
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"Failed to load InsightFace model: {e}")
            self.app = None
            
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.match_threshold = self.config.get('access_control', {}).get('face_match_threshold', 0.6)
        except Exception as e:
            print(f"Error loading config in SmartAttendance: {e}")
            self.match_threshold = 0.6
            
        self.enrolled_faces = {}
        self.enrollment_dir = "data/enrolled_faces"
        os.makedirs(self.enrollment_dir, exist_ok=True)
        self.load_enrolled_faces()

    def enroll_face(self, person_id, image_path):
        """Helper to enroll a face from a photo on disk."""
        if self.app is None: return False
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {image_path}")
            return False
            
        faces = self.app.get(img)
        if not faces:
            print(f"No face detected in {image_path}")
            return False
            
        embedding = faces[0].embedding
        np.save(os.path.join(self.enrollment_dir, f"{person_id}.npy"), embedding)
        self.enrolled_faces[person_id] = embedding
        print(f"Successfully enrolled {person_id}")
        return True

    def load_enrolled_faces(self):
        """Loads all .npy embeddings from the enrollment directory."""
        for filename in os.listdir(self.enrollment_dir):
            if filename.endswith(".npy"):
                person_id = filename.replace(".npy", "")
                try:
                    emb = np.load(os.path.join(self.enrollment_dir, filename))
                    self.enrolled_faces[person_id] = emb
                except Exception as e:
                    print(f"Failed to load embedding {filename}: {e}")

    def detect_faces(self, frame):
        """
        Returns a list of InsightFace objects containing bounding boxes and embeddings.
        This modifies the return signature slightly to avoid double-detection.
        """
        if self.app is None: return []
        try:
            faces = self.app.get(frame)
            return faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []

    def generate_embedding(self, face_obj):
        """
        Extracts the embedding from the insightface object.
        Replaces the old dummy generator.
        """
        if face_obj is None:
            return None
        return face_obj.embedding

    def match_face(self, embedding):
        """
        Matches an embedding against the enrolled faces using cosine similarity.
        Cosine distance: 0 is exact match, 2 is exact opposite. 
        Similarity = 1 - distance.
        """
        if embedding is None or len(self.enrolled_faces) == 0:
            return "UNKNOWN", 0.0
            
        best_match = "UNKNOWN"
        best_sim = -1.0
        
        for person_id, enrolled_emb in self.enrolled_faces.items():
            # Calculate cosine distance (scipy)
            dist = cosine(embedding, enrolled_emb)
            sim = 1.0 - dist
            
            if sim > best_sim:
                best_sim = sim
                best_match = person_id
                
        if best_sim >= self.match_threshold:
            return best_match, float(best_sim)
            
        return "UNKNOWN", 0.0
