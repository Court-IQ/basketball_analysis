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

    def smooth_keypoints(self, court_keypoints, window_size=5):
        """
        Smooths keypoint positions across frames using a rolling average, to reduce
        frame-to-frame jitter in the tactical view / minimap.

        Args:
            court_keypoints (list): List of per-frame Keypoints objects.
            window_size (int): Number of frames to average over (must be odd for symmetry).

        Returns:
            list: The same list, with each frame's x/y positions smoothed in place.
        """
        num_frames = len(court_keypoints)
        half_window = window_size // 2

        for i in range(num_frames):
            start = max(0, i - half_window)
            end = min(num_frames, i + half_window + 1)

            xy_stack = []
            for j in range(start, end):
                kp = court_keypoints[j]
                if kp is not None and kp.data.shape[1] > 0:
                    xy_stack.append(kp.data[..., :2].cpu().numpy())

            if len(xy_stack) == 0:
                continue

            avg_xy = np.mean(np.stack(xy_stack, axis=0), axis=0)

            device = court_keypoints[i].data.device
            new_data = court_keypoints[i].data.clone()
            new_data[..., 0] = torch.from_numpy(avg_xy[..., 0]).to(device)
            new_data[..., 1] = torch.from_numpy(avg_xy[..., 1]).to(device)
            court_keypoints[i].data = new_data

        return court_keypoints