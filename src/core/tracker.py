import math

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {} # dict of id: (centroid, consecutive_frames_disappeared)
        self.max_disappeared = max_disappeared

    def update(self, rects):
        """
        rects: list of bounding boxes [(start_x, start_y, end_x, end_y), ...]
        Returns dict of updated object IDs and their centroids.
        """
        if len(rects) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id] = (self.objects[object_id][0], self.objects[object_id][1] + 1)
                if self.objects[object_id][1] > self.max_disappeared:
                    del self.objects[object_id]
            return {obj_id: centroid for obj_id, (centroid, _) in self.objects.items()}

        input_centroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.objects[self.next_object_id] = (input_centroids[i], 0)
                self.next_object_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id][0] for obj_id in object_ids]

            # Very naive assignment for simplicity: 
            # In a real scenario, use scipy.optimize.linear_sum_assignment (Hungarian Algorithm)
            used_rows = set()
            used_cols = set()
            
            # This is a highly simplified nearest neighbor assignment
            for i, (cX, cY) in enumerate(input_centroids):
                min_dist = float('inf')
                best_obj_id = -1
                
                for obj_id in object_ids:
                    if obj_id in used_cols: continue
                    oX, oY = self.objects[obj_id][0]
                    dist = math.hypot(cX - oX, cY - oY)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_obj_id = obj_id
                        
                if best_obj_id != -1 and min_dist < 100: # Threshold for matching
                    self.objects[best_obj_id] = ((cX, cY), 0)
                    used_cols.add(best_obj_id)
                else:
                    self.objects[self.next_object_id] = ((cX, cY), 0)
                    self.next_object_id += 1

            for obj_id in object_ids:
                if obj_id not in used_cols:
                    self.objects[obj_id] = (self.objects[obj_id][0], self.objects[obj_id][1] + 1)
                    if self.objects[obj_id][1] > self.max_disappeared:
                        del self.objects[obj_id]

        return {obj_id: centroid for obj_id, (centroid, _) in self.objects.items()}
