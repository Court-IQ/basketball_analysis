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

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.65)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_names_inv['Ball']:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    # Reject degenerate boxes
                    if width <= 0 or height <= 0:
                        continue

                    # Ball should be roughly square (round object) - reject elongated
                    # boxes, which are more likely hands, shoes, or misclassified objects
                    aspect_ratio = width / height
                    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                        continue

                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}
                print(f"frame {frame_num}: ball at {chosen_bbox}, conf={max_confidence:.2f}")

        save_stub(stub_path,tracks)
        
        return tracks

    def remove_wrong_detections(self,ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        
        maximum_allowed_distance = 40
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
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

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def filter_ball_near_player_heads(self, ball_positions, player_tracks, head_height_fraction=0.3):
        """
        Reject ball detections that fall inside a player's head region, since these
        are almost always the model mistaking a player's head for the ball.
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