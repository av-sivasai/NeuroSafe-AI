"""
Video Processing Utility Module
================================
Handles video file loading, frame extraction, and preprocessing
for the Neuro-Symbolic AI Safety Inspector.

Functions:
    - extract_frames: Extract frames from a video file at a given interval
    - get_video_info: Retrieve metadata (FPS, resolution, duration)
    - resize_frame: Resize a frame while keeping aspect ratio
"""

import cv2
import numpy as np
import tempfile
import os
from datetime import timedelta


def get_video_info(video_path: str) -> dict:
    """
    Retrieve metadata about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with keys: fps, width, height, total_frames, duration_seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_seconds"] = (
        info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    )
    cap.release()
    return info


def extract_frames(video_path: str, frame_interval: int = 1):
    """
    Generator that yields frames from a video at a given interval.

    Args:
        video_path: Path to the video file.
        frame_interval: Yield every Nth frame (1 = every frame).

    Yields:
        Tuple of (frame_number, frame_bgr_image, timestamp_str).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Calculate human-readable timestamp
            seconds = frame_count / fps if fps > 0 else 0
            timestamp = str(timedelta(seconds=seconds))
            # Trim microseconds for cleaner display
            if "." in timestamp:
                timestamp = timestamp[: timestamp.index(".") + 3]
            yield frame_count, frame, timestamp

        frame_count += 1

    cap.release()


def resize_frame(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Resize a frame to fit within max_width while preserving aspect ratio.

    Args:
        frame: Input BGR image (numpy array).
        max_width: Maximum width in pixels.

    Returns:
        Resized frame as numpy array.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_temp_video(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to a temporary location.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Path to the saved temporary file.
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    return tfile.name


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"
