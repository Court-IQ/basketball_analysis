from ultralytics import YOLO
import numpy as np
import sys
sys.path.append('../')
from utils import read_stub, save_stub


class CourtKeypointDetectorV2:
    """
    Detects named court features (corners, elbows, blocks, etc.) using an
    object detection model, and matches each detection to a fixed real-world
    court coordinate based on its position within the frame.

    Unlike a numbered keypoint/pose model, this model outputs named classes
    (e.g. "elbow", "court_corner") with zero or more instances per frame.
    Since a real court has multiple physical instances of some features
    (4 corners, 4 elbows, etc.), this class disambiguates *which* instance
    is which using relative image position (leftmost = smaller court x,
    topmost = smaller court y), assuming a single static broadcast camera
    per clip.

    Output format: a list (one entry per frame) of dicts mapping a fixed
    point_id (see REAL_WORLD_POINTS) to that point's (x, y) pixel position
    in the frame, for whichever points were confidently detected that frame.
    """

    REAL_WORLD_POINTS = {
        "court_corner_TL": (0.0, 0.0),
        "court_corner_BL": (0.0, 15.0),
        "court_corner_TR": (28.0, 0.0),
        "court_corner_BR": (28.0, 15.0),
        "corner_three_TL": (0.0, 0.91),
        "corner_three_BL": (0.0, 14.1),
        "corner_three_TR": (28.0, 0.91),
        "corner_three_BR": (28.0, 14.1),
        "block_TL": (0.0, 5.18),
        "block_BL": (0.0, 9.82),
        "block_TR": (28.0, 5.18),
        "block_BR": (28.0, 9.82),
        "elbow_TL": (5.79, 5.18),
        "elbow_BL": (5.79, 9.82),
        "elbow_TR": (22.21, 5.18),
        "elbow_BR": (22.21, 9.82),
        "apex_L": (5.79, 7.5),
        "apex_R": (22.21, 7.5),
        "halfcourt_sideline_top": (14.0, 0.0),
        "halfcourt_sideline_bottom": (14.0, 15.0),
        "center_court": (14.0, 7.5),
    }

    CLASS_CONFIG = {
        "court_corner": ("quad", ["court_corner_TL", "court_corner_BL", "court_corner_TR", "court_corner_BR"]),
        "corner_three": ("quad", ["corner_three_TL", "corner_three_BL", "corner_three_TR", "corner_three_BR"]),
        "block": ("quad", ["block_TL", "block_BL", "block_TR", "block_BR"]),
        "elbow": ("quad", ["elbow_TL", "elbow_BL", "elbow_TR", "elbow_BR"]),
        "three_pointer_apex": ("pair_lr", ["apex_L", "apex_R"]),
        "top_of_key": ("pair_lr", ["apex_L", "apex_R"]),
        "halfcourt_sideline": ("pair_tb", ["halfcourt_sideline_top", "halfcourt_sideline_bottom"]),
        "center_court": ("single", ["center_court"]),
        "center_circle": ("single", ["center_court"]),
    }

    def __init__(self, model_path, confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=self.confidence)
            detections += detections_batch
        return detections

    def _match_quad(self, boxes, point_ids):
        if len(boxes) == 0:
            return {}
        boxes = sorted(boxes, key=lambda b: b[0])
        matches = {}
        if len(boxes) <= 2:
            local = sorted(boxes, key=lambda b: b[1])
            ids = [point_ids[0], point_ids[1]]
            for box, pid in zip(local, ids):
                matches[pid] = (box[0], box[1])
        else:
            mid = len(boxes) // 2
            left = sorted(boxes[:mid], key=lambda b: b[1])
            right = sorted(boxes[mid:], key=lambda b: b[1])
            left_ids = [point_ids[0], point_ids[1]]
            right_ids = [point_ids[2], point_ids[3]]
            for box, pid in zip(left, left_ids):
                matches[pid] = (box[0], box[1])
            for box, pid in zip(right, right_ids):
                matches[pid] = (box[0], box[1])
        return matches

    def _match_pair_lr(self, boxes, point_ids):
        if len(boxes) == 0:
            return {}
        boxes = sorted(boxes, key=lambda b: b[0])
        matches = {}
        if len(boxes) == 1:
            matches[point_ids[0]] = (boxes[0][0], boxes[0][1])
        else:
            matches[point_ids[0]] = (boxes[0][0], boxes[0][1])
            matches[point_ids[1]] = (boxes[-1][0], boxes[-1][1])
        return matches

    def _match_pair_tb(self, boxes, point_ids):
        if len(boxes) == 0:
            return {}
        boxes = sorted(boxes, key=lambda b: b[1])
        matches = {}
        if len(boxes) == 1:
            matches[point_ids[0]] = (boxes[0][0], boxes[0][1])
        else:
            matches[point_ids[0]] = (boxes[0][0], boxes[0][1])
            matches[point_ids[1]] = (boxes[-1][0], boxes[-1][1])
        return matches

    def _match_single(self, boxes, point_ids):
        if len(boxes) == 0:
            return {}
        best = max(boxes, key=lambda b: b[2])
        return {point_ids[0]: (best[0], best[1])}

    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None):
        court_keypoints = read_stub(read_from_stub, stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                return court_keypoints

        detections = self.detect_frames(frames)
        court_keypoints = []

        for detection in detections:
            names = detection.names
            per_class_boxes = {}

            for box in detection.boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                per_class_boxes.setdefault(cls_name, []).append((cx, cy, conf))

            frame_matches = {}
            for cls_name, boxes in per_class_boxes.items():
                if cls_name not in self.CLASS_CONFIG:
                    continue
                kind, point_ids = self.CLASS_CONFIG[cls_name]
                if kind == "quad":
                    matches = self._match_quad(boxes, point_ids)
                elif kind == "pair_lr":
                    matches = self._match_pair_lr(boxes, point_ids)
                elif kind == "pair_tb":
                    matches = self._match_pair_tb(boxes, point_ids)
                else:
                    matches = self._match_single(boxes, point_ids)

                for pid, coords in matches.items():
                    if pid not in frame_matches:
                        frame_matches[pid] = coords

            court_keypoints.append(frame_matches)

        save_stub(stub_path, court_keypoints)
        return court_keypoints

    def get_real_world_point(self, point_id):
        return self.REAL_WORLD_POINTS[point_id]