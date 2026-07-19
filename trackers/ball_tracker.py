from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path) 

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.65)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, max_jump_distance=80):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Picks whichever detection is closest to the last known ball position, instead
        of always trusting the highest-confidence detection. Also includes a drift
        correction: if some far-away detection keeps showing up consistently for
        several frames straight, it's probably the real ball and we've locked onto
        the wrong object - so we re-anchor to it.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]
        last_known_center = None
        frames_since_last_seen = 0
        drift_candidate_center = None
        drift_streak = 0
        drift_confirm_frames = 8

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})

            candidates = []
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                class_name = cls_names.get(cls_id, "").lower()
                if class_name not in ("ball", "basketball"):
                    continue

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width <= 0 or height <= 0:
                    continue

                aspect_ratio = width / height
                if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                    continue

                center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                candidates.append({"bbox": bbox, "confidence": confidence, "center": center})

            chosen_bbox = None

            if len(candidates) > 0:
                if last_known_center is not None:
                    allowed_distance = max_jump_distance * max(1, frames_since_last_seen)

                    nearby_candidates = []
                    for c in candidates:
                        dist = np.linalg.norm(np.array(c["center"]) - np.array(last_known_center))
                        if dist <= allowed_distance:
                            nearby_candidates.append(c)

                    if len(nearby_candidates) > 0:
                        best = max(nearby_candidates, key=lambda c: c["confidence"])
                        chosen_bbox = best["bbox"]

                    # --- Drift correction ---
                    far_candidates = [c for c in candidates if c not in nearby_candidates]
                    if len(far_candidates) > 0:
                        far_best = max(far_candidates, key=lambda c: c["confidence"])
                        if drift_candidate_center is not None:
                            drift_dist = np.linalg.norm(np.array(far_best["center"]) - np.array(drift_candidate_center))
                        else:
                            drift_dist = None

                        if drift_dist is not None and drift_dist <= max_jump_distance:
                            drift_streak += 1
                        else:
                            drift_streak = 1
                        drift_candidate_center = far_best["center"]

                        if drift_streak >= drift_confirm_frames:
                            chosen_bbox = far_best["bbox"]
                            drift_streak = 0
                            drift_candidate_center = None
                    else:
                        drift_streak = 0
                        drift_candidate_center = None
                else:
                    best = max(candidates, key=lambda c: c["confidence"])
                    chosen_bbox = best["bbox"]

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}
                last_known_center = ((chosen_bbox[0]+chosen_bbox[2])/2, (chosen_bbox[1]+chosen_bbox[3])/2)
                frames_since_last_seen = 0
            else:
                frames_since_last_seen += 1

        save_stub(stub_path,tracks)
        
        return tracks

    def filter_ball_near_player_heads(self, ball_positions, player_tracks, head_height_fraction=0.3):
        """
        Reject ball detections that fall inside a player's head region.
        """
        for frame_num in range(min(len(ball_positions), len(player_tracks))):
            ball_box = ball_positions[frame_num].get(1, {}).get('bbox', [])
            if len(ball_box) == 0:
                continue

            ball_center_x = (ball_box[0] + ball_box[2]) / 2
            ball_center_y = (ball_box[1] + ball_box[3]) / 2

            for player_id, player_data in player_tracks[frame_num].items():
                p_bbox = player_data['bbox']
                p_x1, p_y1, p_x2, p_y2 = p_bbox
                head_region_bottom = p_y1 + (p_y2 - p_y1) * head_height_fraction

                if p_x1 <= ball_center_x <= p_x2 and p_y1 <= ball_center_y <= head_region_bottom:
                    ball_positions[frame_num] = {}
                    break

        return ball_positions

    def remove_wrong_detections(self,ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.
        """
        maximum_allowed_distance = 40
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions