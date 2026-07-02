"""
Person Tracker
==============
IoU + centroid-based tracker that assigns a random, persistent
alphanumeric ID to each detected person and maintains it across
frames.  IDs survive brief occlusions (up to MAX_MISSING frames).

Public API:
    tracker = PersonTracker()
    tracked = tracker.update(person_boxes)   # list of (id, x1, y1, x2, y2)
"""

import random
import string
import numpy as np
from typing import List, Tuple, Dict

# How many consecutive frames a person can be absent before we
# consider them gone (and their ID retired).
MAX_MISSING = 30


def _random_id(length: int = 6) -> str:
    """Generate a random alphanumeric ID like 'A3F9K2'."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def _iou(boxA: Tuple, boxB: Tuple) -> float:
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """
    Lightweight IoU-based tracker for person bounding boxes.

    Attributes:
        tracks  : dict {person_id: {"box": (x1,y1,x2,y2), "missing": int}}
        IOU_THRESH: minimum IoU to consider two boxes the same person
    """

    IOU_THRESH = 0.25   # lower than typical to handle occlusion / dim light

    def __init__(self):
        self.tracks: Dict[str, dict] = {}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def update(
        self, person_boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Match new bounding boxes to existing tracks.

        Args:
            person_boxes: list of (x1, y1, x2, y2) from current frame.

        Returns:
            list of (person_id, x1, y1, x2, y2) for every active track.
        """
        # ── 1. Build cost matrix (IoU between every existing track
        #        and every new detection) ──
        track_ids = list(self.tracks.keys())
        matched_track_ids = set()
        matched_det_indices = set()

        if track_ids and person_boxes:
            iou_matrix = np.zeros((len(track_ids), len(person_boxes)))
            for ti, tid in enumerate(track_ids):
                for di, det_box in enumerate(person_boxes):
                    iou_matrix[ti, di] = _iou(self.tracks[tid]["box"], det_box)

            # Greedy matching: pick highest IoU pairs first
            flat_indices = np.argsort(-iou_matrix, axis=None)
            for idx in flat_indices:
                ti, di = divmod(int(idx), len(person_boxes))
                if ti in matched_track_ids or di in matched_det_indices:
                    continue
                if iou_matrix[ti, di] >= self.IOU_THRESH:
                    tid = track_ids[ti]
                    self.tracks[tid]["box"] = person_boxes[di]
                    self.tracks[tid]["missing"] = 0
                    matched_track_ids.add(ti)
                    matched_det_indices.add(di)

        # ── 2. Create new tracks for unmatched detections ──
        for di, det_box in enumerate(person_boxes):
            if di not in matched_det_indices:
                new_id = _random_id()
                # Ensure uniqueness
                while new_id in self.tracks:
                    new_id = _random_id()
                self.tracks[new_id] = {"box": det_box, "missing": 0}

        # ── 3. Age unmatched existing tracks ──
        for ti, tid in enumerate(track_ids):
            if ti not in matched_track_ids:
                self.tracks[tid]["missing"] += 1

        # ── 4. Prune stale tracks ──
        stale = [tid for tid, t in self.tracks.items() if t["missing"] > MAX_MISSING]
        for tid in stale:
            del self.tracks[tid]

        # ── 5. Return active (non-missing) tracks ──
        result = []
        for tid, t in self.tracks.items():
            if t["missing"] == 0:
                x1, y1, x2, y2 = t["box"]
                result.append((tid, x1, y1, x2, y2))

        return result
