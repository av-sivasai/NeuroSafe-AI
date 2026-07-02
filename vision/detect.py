"""
Computer Vision Detection Module  ─  Maximum Accuracy Edition
==============================================================
Strategy stack for highest-possible person recall:

  1. MODEL
     • Auto-selects yolov8m.pt (medium, 25M params) over yolov8n.pt.
       YOLOv8m has ~3× more capacity and significantly better recall on
       partially-occluded / small / distant persons.
     • Falls back to the provided model_path if neither is cached.

  2. PERSON-SPECIFIC CONFIDENCE
     • Person class uses a *lower* confidence threshold (PERSON_CONF)
       than other classes (0.25 vs. 0.45).  This avoids missing low-
       confidence partial detections at the cost of minor false positives
       (filtered downstream by the tracker's persistence logic).

  3. TEST-TIME AUGMENTATION (TTA)
     • YOLO's built-in augment=True flag runs inference at multiple
       scales + flips and fuses the results — free accuracy boost.

  4. MULTI-SCALE PASS
     • A second pass at 1.3× upscaled frame catches small/distant persons
       that fall below the detector's native resolution threshold.
       Detections are re-projected to original coordinates.

  5. LOW-LIGHT PREPROCESSING PIPELINE
       ┌─ Gamma correction (auto-exposes dark frames)
       ├─ Bilateral filter   (edge-preserving denoise)
       ├─ CLAHE on L channel (local contrast)
       └─ Unsharp mask       (restore edge crispness after smoothing)

  6. PER-PERSON GEAR ATTRIBUTION
     • Gear bbox overlap is checked against expanded person boxes
       (+5% padding) to tolerate slight misalignment.

  7. DEDUPLICATION
     • Detections from multiple passes are merged with IoU-based
       deduplication (threshold 0.50) so the same person is not
       counted twice.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Dict, List, Optional
import os

from tracking.tracker import PersonTracker


# ─── Class IDs ────────────────────────────────────────────────────
PERSON_CLASS_ID  = 0
HELMET_PROXY_IDS = {25, 28}         # umbrella, suitcase
VEST_PROXY_IDS   = {24, 26, 27}     # backpack, handbag, tie

# ─── Thresholds ────────────────────────────────────────────────────
PERSON_CONF      = 0.25   # low threshold → maximum recall for persons
OTHER_CONF       = 0.45   # higher bar for non-person classes
GEAR_OVERLAP_THR = 0.08   # gear box must overlap >= 8% of person box
DEDUP_IOU_THR    = 0.50   # IoU above which two person boxes are the same

# ─── Model preference ─────────────────────────────────────────────
# Try medium model first (much better accuracy), fall back to nano.
PREFERRED_MODELS = ["yolov8m.pt", "yolov8n.pt"]


# ══════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════

def _auto_gamma(frame: np.ndarray) -> np.ndarray:
    """
    Automatically brighten dark frames using adaptive gamma correction.
    Gamma is computed from the mean luminance of the frame:
      • Very dark  → strong lift  (gamma ≈ 0.4)
      • Well-lit   → no change    (gamma ≈ 1.0)
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean  = np.mean(gray) / 255.0          # normalised mean brightness
    # Desired target brightness ≈ 0.45; gamma = log(target)/log(mean)
    if mean < 0.01:
        mean = 0.01
    gamma = np.log(0.45) / np.log(mean)
    gamma = float(np.clip(gamma, 0.35, 2.5))   # cap to avoid runaway

    # Build lookup table (much faster than per-pixel math)
    inv_gamma = 1.0 / gamma
    table     = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, table)


def _unsharp_mask(
    frame: np.ndarray, kernel: int = 5, strength: float = 1.4
) -> np.ndarray:
    """Unsharp masking to restore edge crispness after blur."""
    blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
    return cv2.addWeighted(frame, strength, blurred, -(strength - 1.0), 0)


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """
    Full low-light enhancement pipeline:
      1. Auto-gamma  — expose dark areas
      2. Bilateral   — denoise without destroying edges
      3. CLAHE       — local contrast on L channel (LAB)
      4. Unsharp     — sharpen back for detector edge response
    """
    frame = _auto_gamma(frame)
    frame = cv2.bilateralFilter(frame, d=7, sigmaColor=60, sigmaSpace=60)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    frame = _unsharp_mask(frame, kernel=5, strength=1.3)
    return frame


# ══════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════

def _iou(a: Tuple, b: Tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _overlap_ratio(inner: Tuple, outer: Tuple) -> float:
    """Fraction of *inner* that overlaps with *outer*."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (inner[2]-inner[0]) * (inner[3]-inner[1]))
    return inter / area


def _expand_box(
    box: Tuple, pad: float, w: int, h: int
) -> Tuple[int, int, int, int]:
    """Expand a person box by `pad` fraction, clamped to frame bounds."""
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * pad)
    dy = int((y2 - y1) * pad)
    return (
        max(0, x1 - dx), max(0, y1 - dy),
        min(w, x2 + dx), min(h, y2 + dy),
    )


def _deduplicate(
    boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Remove duplicate / heavily-overlapping boxes.
    Keeps the larger box when two overlap > DEDUP_IOU_THR.
    """
    if not boxes:
        return boxes
    # Sort by area descending (keep biggest)
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    kept = []
    for box in boxes:
        if all(_iou(box, k) < DEDUP_IOU_THR for k in kept):
            kept.append(box)
    return kept


# ══════════════════════════════════════════════════════════════════
# SafetyDetector
# ══════════════════════════════════════════════════════════════════

class SafetyDetector:
    """
    Maximum-accuracy workplace safety detector.

    Detection stack:
      • YOLOv8m (preferred) / YOLOv8n (fallback)
      • TTA (augment=True)
      • Multi-scale second pass at 1.3× upscale
      • Full low-light preprocessing pipeline
      • Person-specific low confidence threshold
      • IoU deduplication across passes
      • Per-person gear attribution with expanded boxes
      • PersonTracker for persistent random IDs

    Args:
        model_path   : override model weights (default: auto-select m→n)
        confidence   : base confidence for non-person classes
        simulate_gear: simulate helmet/vest for demo
        enhance_light: enable the low-light pipeline
        use_tta      : enable test-time augmentation
        multiscale   : run a second pass at higher resolution
    """

    def __init__(
        self,
        model_path:    str   = "auto",
        confidence:    float = OTHER_CONF,
        simulate_gear: bool  = True,
        enhance_light: bool  = True,
        use_tta:       bool  = True,
        multiscale:    bool  = True,
    ):
        self.confidence    = confidence
        self.simulate_gear = simulate_gear
        self.enhance_light = enhance_light
        self.use_tta       = use_tta
        self.multiscale    = multiscale
        self._tracker      = PersonTracker()
        self._frame_no     = 0
        self._sim_cache:   Dict[str, Tuple[bool, bool]] = {}

        self.model = self._load_model(model_path)

    # ── model loading ──────────────────────────────────────────────

    def _load_model(self, model_path: str) -> YOLO:
        if model_path != "auto":
            print(f"[SafetyDetector] Loading {model_path}")
            return YOLO(model_path)

        for name in PREFERRED_MODELS:
            try:
                print(f"[SafetyDetector] Trying {name} …")
                m = YOLO(name)   # downloads automatically if not cached
                print(f"[SafetyDetector] Loaded {name}")
                return m
            except Exception as e:
                print(f"[SafetyDetector] {name} failed: {e}")

        raise RuntimeError("Could not load any YOLOv8 model.")

    # ── public API ─────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray, List[Dict]]:
        """
        Run the full detection-tracking pipeline on one frame.

        Returns:
          persons    : [{"id":str, "box":(x1,y1,x2,y2),
                         "helmet":bool, "vest":bool}, ...]
          annotated  : BGR frame with overlays
          raw_dets   : raw YOLO detections [{label, confidence, bbox}, ...]
        """
        self._frame_no += 1
        h, w = frame.shape[:2]

        # ── 1. Preprocessing ──
        enhanced = enhance_low_light(frame) if self.enhance_light else frame.copy()

        # ── 2. Primary inference (TTA enabled) ──
        person_boxes, gear_boxes, raw_dets = self._run_inference(
            enhanced, w, h, augment=self.use_tta
        )

        # ── 3. Multi-scale second pass ──
        if self.multiscale:
            scale      = 1.30
            big        = cv2.resize(
                enhanced,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            pb2, gb2, rd2 = self._run_inference(big, int(w*scale), int(h*scale),
                                                 augment=False)
            # Re-project coordinates back to original resolution
            inv = 1.0 / scale
            pb2 = [
                (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
                for x1, y1, x2, y2 in pb2
            ]
            for gt in ("helmet", "vest"):
                gb2[gt] = [
                    (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
                    for x1, y1, x2, y2 in gb2[gt]
                ]
            person_boxes += pb2
            gear_boxes["helmet"] += gb2["helmet"]
            gear_boxes["vest"]   += gb2["vest"]
            raw_dets             += rd2

        # ── 4. Deduplicate person boxes ──
        person_boxes = _deduplicate(person_boxes)

        # ── 5. Track → persistent IDs ──
        tracked = self._tracker.update(person_boxes)

        # ── 6. Per-person gear attribution ──
        persons: List[Dict] = []
        for pid, x1, y1, x2, y2 in tracked:
            p_box = (x1, y1, x2, y2)

            if self.simulate_gear:
                helmet, vest = self._simulate_gear_for(pid)
            else:
                # Expand box slightly for more tolerant overlap check
                exp = _expand_box(p_box, 0.05, w, h)
                helmet = any(
                    _overlap_ratio(g, exp) >= GEAR_OVERLAP_THR
                    for g in gear_boxes["helmet"]
                )
                vest = any(
                    _overlap_ratio(g, exp) >= GEAR_OVERLAP_THR
                    for g in gear_boxes["vest"]
                )

            persons.append({"id": pid, "box": p_box,
                            "helmet": helmet, "vest": vest})

        # ── 7. Annotate ──
        annotated = self._annotate(frame.copy(), persons, raw_dets)
        return persons, annotated, raw_dets

    # ── inference helper ───────────────────────────────────────────

    def _run_inference(
        self, img: np.ndarray, img_w: int, img_h: int, augment: bool
    ) -> Tuple[List[Tuple], Dict[str, List[Tuple]], List[Dict]]:
        """
        Run YOLO on `img`.  Returns separate lists for person boxes,
        gear boxes (keyed by type), and raw detections.
        Uses PERSON_CONF as the minimum threshold so no person is
        missed purely because of a confidence gap.
        """
        results  = self.model(
            img,
            conf=PERSON_CONF,       # low enough to catch faint persons
            iou=0.45,               # NMS IoU — allows closer boxes
            augment=augment,
            verbose=False,
            classes=[               # only care about these COCO classes
                PERSON_CLASS_ID,
                *HELMET_PROXY_IDS,
                *VEST_PROXY_IDS,
            ],
        )[0]

        person_boxes: List[Tuple]            = []
        gear_boxes:   Dict[str, List[Tuple]] = {"helmet": [], "vest": []}
        raw_dets:     List[Dict]             = []

        seen: set = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            # Apply per-class confidence gate
            min_conf = PERSON_CONF if cls_id == PERSON_CLASS_ID else OTHER_CONF
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = self.model.names[cls_id]

            # Grid-based dedup within single pass
            key = (round(x1, -1), round(y1, -1), cls_id)
            if key in seen:
                continue
            seen.add(key)

            raw_dets.append({
                "label": label, "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })

            if cls_id == PERSON_CLASS_ID:
                # Sanity-check: person box must have reasonable area
                area = (x2-x1) * (y2-y1)
                if area > 200:          # skip tiny blobs (< ~14×14 px)
                    person_boxes.append((x1, y1, x2, y2))
            elif cls_id in HELMET_PROXY_IDS:
                gear_boxes["helmet"].append((x1, y1, x2, y2))
            elif cls_id in VEST_PROXY_IDS:
                gear_boxes["vest"].append((x1, y1, x2, y2))

        return person_boxes, gear_boxes, raw_dets

    # ── simulation ─────────────────────────────────────────────────

    def _simulate_gear_for(self, person_id: str) -> Tuple[bool, bool]:
        """Stable per-ID simulated gear state (cycles through 4 scenarios)."""
        if person_id not in self._sim_cache:
            bucket = len(self._sim_cache) % 4
            scenarios = [
                (True,  True),    # compliant
                (False, True),    # no helmet
                (True,  False),   # no vest
                (False, False),   # no equipment
            ]
            self._sim_cache[person_id] = scenarios[bucket]
        return self._sim_cache[person_id]

    # ── annotation ─────────────────────────────────────────────────

    def _annotate(
        self,
        frame:    np.ndarray,
        persons:  List[Dict],
        raw_dets: List[Dict],
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Draw semi-transparent overlay panel for legend
        overlay = frame.copy()

        for p in persons:
            x1, y1, x2, y2 = p["box"]
            compliant = p["helmet"] and p["vest"]
            color     = (40, 210, 40) if compliant else (20, 40, 230)

            # Filled transparent rect
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, 0.05, frame, 0.95, 0)  # 5% tint

        for p in persons:
            x1, y1, x2, y2 = p["box"]
            compliant = p["helmet"] and p["vest"]
            color     = (40, 210, 40) if compliant else (20, 40, 230)

            # Solid bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # ── ID pill ──
            label = f"ID: {p['id']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            pill_y = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, pill_y), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

            # ── Gear icons ──
            gy = y1 + 22
            h_txt = ("H\u2713" if p["helmet"] else "H\u2717")
            v_txt = ("V\u2713" if p["vest"]   else "V\u2717")
            hc = (40, 210, 40) if p["helmet"] else (20, 40, 230)
            vc = (40, 210, 40) if p["vest"]   else (20, 40, 230)
            cv2.putText(frame, h_txt, (x1+4,  gy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hc, 2)
            cv2.putText(frame, v_txt, (x1+46, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, vc, 2)

        # ── Top-left HUD ──
        hud_lines = [
            f"Persons: {len(persons)}",
            f"Frame:   {self._frame_no}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(
                frame, line, (10, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 210, 0), 2,
            )

        return frame
