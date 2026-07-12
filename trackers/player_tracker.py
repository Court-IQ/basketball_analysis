from ultralytics import YOLO
import supervision as sv
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.
    """
    def __init__(self, model_path, frame_rate=30):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
            frame_rate (int): Frame rate of the source video, used for track buffer timing.
        """
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=120,
            minimum_matching_threshold=0.75,
            frame_rate=frame_rate
        )

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.64, iou=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
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

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {"bbox":bbox}
        
        save_stub(stub_path,tracks)
        return tracks

    def merge_broken_tracks(self, tracks, max_frame_gap=15, max_distance=150):
        """
        Merge player IDs that disappear and reappear nearby within a short window,
        treating them as the same physical player instead of a new one.

        Args:
            tracks (list): List of per-frame dicts mapping track_id -> {"bbox": bbox}.
            max_frame_gap (int): Max frames a player can be missing and still get re-linked.
            max_distance (float): Max pixel distance between last-seen and reappeared position.

        Returns:
            list: Same structure as input, with fragmented IDs merged together.
        """
        last_seen = {}
        id_remap = {}

        for frame_num, frame_tracks in enumerate(tracks):
            for track_id, data in list(frame_tracks.items()):
                bbox = data['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                real_id = id_remap.get(track_id, track_id)

                if track_id not in last_seen:
                    for old_id, (old_frame, old_center) in last_seen.items():
                        if old_id == real_id:
                            continue
                        frame_gap = frame_num - old_frame
                        if 0 < frame_gap <= max_frame_gap:
                            dist = ((center[0]-old_center[0])**2 + (center[1]-old_center[1])**2) ** 0.5
                            if dist <= max_distance:
                                id_remap[track_id] = old_id
                                real_id = old_id
                                break

                last_seen[real_id] = (frame_num, center)

        new_tracks = []
        for frame_tracks in tracks:
            new_frame = {}
            for track_id, data in frame_tracks.items():
                real_id = id_remap.get(track_id, track_id)
                new_frame[real_id] = data
            new_tracks.append(new_frame)

        return new_tracks