import torch
import cv2
import numpy as np
from collections import OrderedDict
from dedodev2 import dedode_detector_L, dedode_descriptor_B
from dedodev2.dual_softmax_matcher import DualSoftMaxMatcher
from dedodev2.utils import to_pixel_coords


class FeatureTracker:
    def __init__(self, buffer_size=10, max_disappeared=3, min_points=100,  comp_thres=0.75):
        self.buffer_size = buffer_size
        self.max_disappeared = max_disappeared
        self.min_points = min_points
        self.comp_thres = comp_thres

        self.detector = dedode_detector_L()
        self.descriptor = dedode_descriptor_B()
        self.matcher = DualSoftMaxMatcher()

        self.tracked_kps = OrderedDict()
        self.tracked_descs = OrderedDict()
        self.tracks = OrderedDict()
        self.disappeared = OrderedDict()
        self.colors = OrderedDict()
        self.next_object_id = 0

    def update(self, frame):
        # Detect keypoints
        detections = self.detector.detect(frame, num_keypoints=1024)
        keypoints = detections["keypoints"]

        # Describe keypoints
        descriptions = self.descriptor.describe_keypoints(frame, keypoints)["descriptions"]

        # If no keypoints detected, increment missing and return
        if len(keypoints) == 0:
            self.increment_all_missing()
            return False, []

        # If no keypoints are being tracked, register all detected keypoints
        if len(self.tracked_kps) == 0:
            keypoints = to_pixel_coords(keypoints, frame.shape[0], frame.shape[1])
            for kp, desc in zip(keypoints[0], descriptions[0]):
                self.register(kp, desc)
            return False, []

        # Match new keypoints with tracked keypoints
        tracked_descs_array = torch.stack(list(self.tracked_descs.values()))

        indices_1, indices_2, _ = self.matcher.match(
            torch.stack(list(self.tracked_kps.values()))[None],
            tracked_descs_array[None],
            keypoints,
            descriptions,
            normalize=True,
            inv_temp=20,
            threshold=self.comp_thres,
        )

        keypoints = to_pixel_coords(keypoints, frame.shape[0], frame.shape[1])

        # Update matches
        self.update_matches(keypoints[0], descriptions[0], indices_1, indices_2)

        # Increment missing keypoints
        self.increment_missing_kps(indices_1)

        # Register new keypoints if tracked keypoints are below threshold
        if len(self.tracked_kps) < self.min_points:
            self.register_non_visited(keypoints[0], descriptions[0], indices_2)

        # Update tracks
        self.update_tracks()

        return True, indices_1

    def register(self, kp, desc):
        self.tracked_kps[self.next_object_id] = kp
        self.tracked_descs[self.next_object_id] = desc
        self.disappeared[self.next_object_id] = 0
        self.colors[self.next_object_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.tracked_kps[object_id]
        del self.tracked_descs[object_id]
        del self.disappeared[object_id]
        del self.colors[object_id]
        if object_id in self.tracks:
            del self.tracks[object_id]

    def update_matches(self, new_kps, new_descs, indices_1, indices_2):
        for i1, i2 in zip(indices_1, indices_2):
            tracked_id = list(self.tracked_kps.keys())[i1]
            self.tracked_kps[tracked_id] = new_kps[i2]
            self.tracked_descs[tracked_id] = new_descs[i2]
            self.disappeared[tracked_id] = 0

    def update_tracks(self):
        for kp_id, kp in self.tracked_kps.items():
            if kp_id not in self.tracks:
                self.tracks[kp_id] = []
            if len(self.tracks[kp_id]) > self.buffer_size:
                self.tracks[kp_id].pop(0)
            self.tracks[kp_id].append((int(kp[0]), int(kp[1])))

    def register_non_visited(self, new_kps, new_descs, visited_indices):
        visited_set = set(visited_indices.tolist())
        for i, (kp, desc) in enumerate(zip(new_kps, new_descs)):
            if i not in visited_set:
                self.register(kp, desc)

    def increment_missing_kps(self, visited_indices):
        visited_set = set(visited_indices.tolist())
        for object_id in list(self.tracked_kps.keys()):
            if object_id not in visited_set:
                self.increment_missing_kp(object_id)

    def increment_missing_kp(self, object_id):
        self.disappeared[object_id] += 1
        if self.disappeared[object_id] >= self.max_disappeared:
            self.deregister(object_id)

    def increment_all_missing(self):
        for object_id in list(self.tracked_kps.keys()):
            self.increment_missing_kp(object_id)

    def draw_tracks(self, img):
        vis = img.copy()
        for object_id, track in self.tracks.items():
            if len(track) < 2:
                continue
            for i in range(1, len(track)):
                cv2.line(vis, track[i - 1], track[i], self.colors[object_id], 2)
            cv2.circle(vis, track[-1], 3, self.colors[object_id], -1)
        return vis