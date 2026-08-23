import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict() # dict of id: (centroid, rect, consecutive_frames_disappeared)
        self.max_disappeared = max_disappeared

    def update(self, rects):
        """
        rects: list of bounding boxes [(start_x, start_y, end_x, end_y), ...]
        Returns:
            assigned_objects: list of (object_id, rect) corresponding to the input rects
            evicted_ids: set of object_ids that were removed this frame
        """
        evicted_ids = set()
        
        if len(rects) == 0:
            for object_id in list(self.objects.keys()):
                centroid, rect, disappeared = self.objects[object_id]
                self.objects[object_id] = (centroid, rect, disappeared + 1)
                if self.objects[object_id][2] > self.max_disappeared:
                    evicted_ids.add(object_id)
                    del self.objects[object_id]
            return [], evicted_ids

        input_centroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            assigned_objects = []
            for i in range(len(input_centroids)):
                self.objects[self.next_object_id] = (input_centroids[i], rects[i], 0)
                assigned_objects.append((self.next_object_id, rects[i]))
                self.next_object_id += 1
            return assigned_objects, evicted_ids
            
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[obj_id][0] for obj_id in object_ids]

        # Compute cost matrix (Euclidean distance)
        cost_matrix = np.zeros((len(object_centroids), len(input_centroids)))
        for i, (ox, oy) in enumerate(object_centroids):
            for j, (cx, cy) in enumerate(input_centroids):
                cost_matrix[i, j] = math.hypot(ox - cx, oy - cy)

        # Scipy linear_sum_assignment (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        used_rows = set()
        used_cols = set()
        assigned_mapping = {} # maps col (input index) to obj_id

        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] > 100: # Distance threshold
                continue
            
            object_id = object_ids[row]
            self.objects[object_id] = (input_centroids[col], rects[col], 0)
            assigned_mapping[col] = object_id
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched input rects (new objects)
        for col in range(len(input_centroids)):
            if col not in used_cols:
                object_id = self.next_object_id
                self.objects[object_id] = (input_centroids[col], rects[col], 0)
                assigned_mapping[col] = object_id
                self.next_object_id += 1

        # Handle unmatched existing objects (disappeared)
        for row in range(len(object_centroids)):
            if row not in used_rows:
                object_id = object_ids[row]
                centroid, rect, disappeared = self.objects[object_id]
                self.objects[object_id] = (centroid, rect, disappeared + 1)
                if self.objects[object_id][2] > self.max_disappeared:
                    evicted_ids.add(object_id)
                    del self.objects[object_id]

        # Build parallel assigned_objects list
        assigned_objects = []
        for i in range(len(rects)):
            assigned_objects.append((assigned_mapping[i], rects[i]))

        return assigned_objects, evicted_ids
