from ultralytics import YOLO
import supervision as sv
import torch
import numpy as np
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames. 
    It also provides functionality to draw these detected keypoints on the frames.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested, 
        attempts to read previously detected keypoints from a stub file before running the model.
        """
        court_keypoints = read_stub(read_from_stub, stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                return court_keypoints
        
        batch_size = 20
        court_keypoints = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.5)
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)

        court_keypoints = self.smooth_keypoints(court_keypoints, window_size=5)

        save_stub(stub_path, court_keypoints)
        
        return court_keypoints

    def smooth_keypoints(self, court_keypoints, window_size=5, confidence_threshold=0.5):
        """
        Smooths keypoint positions across frames using a rolling average, to reduce
        frame-to-frame jitter in the tactical view / minimap.

        Only averages a keypoint using frames where that specific point was detected
        with high enough confidence - low-confidence/occluded detections are excluded
        so they don't drag the average off the true court line.

        Args:
            court_keypoints (list): List of per-frame Keypoints objects.
            window_size (int): Number of frames to average over (must be odd for symmetry).
            confidence_threshold (float): Minimum per-point confidence to include in averaging.

        Returns:
            list: The same list, with each frame's x/y positions smoothed in place.
        """
        num_frames = len(court_keypoints)
        half_window = window_size // 2

        for i in range(num_frames):
            kp_current = court_keypoints[i]
            if kp_current is None or kp_current.data.shape[1] == 0:
                continue

            start = max(0, i - half_window)
            end = min(num_frames, i + half_window + 1)

            num_points = kp_current.data.shape[1]
            has_conf = kp_current.data.shape[-1] > 2
            device = kp_current.data.device
            new_data = kp_current.data.clone()

            for point_idx in range(num_points):
                xy_stack = []
                for j in range(start, end):
                    kp = court_keypoints[j]
                    if kp is None or kp.data.shape[1] <= point_idx:
                        continue
                    conf = kp.data[0, point_idx, 2].item() if has_conf else 1.0
                    if conf >= confidence_threshold:
                        xy_stack.append(kp.data[0, point_idx, :2].cpu().numpy())

                if len(xy_stack) == 0:
                    continue  # no confident detections in window, leave this point as-is

                avg_xy = np.mean(np.stack(xy_stack, axis=0), axis=0)
                new_data[0, point_idx, 0] = torch.tensor(avg_xy[0], device=device)
                new_data[0, point_idx, 1] = torch.tensor(avg_xy[1], device=device)

            court_keypoints[i].data = new_data

        return court_keypoints