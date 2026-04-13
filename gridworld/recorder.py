"""Video recorder — captures pygame frames to MP4 via OpenCV."""

import os
import numpy as np
import cv2
import pygame


class VideoRecorder:
    """Captures pygame screen to an MP4 file.

    Usage:
        recorder = VideoRecorder("output.mp4", screen, fps=30)
        # in game loop:
        recorder.capture(screen)
        # when done:
        recorder.finish()
    """

    def __init__(self, path: str, screen: pygame.Surface, fps: int = 30,
                 sample_every: int = 1):
        """
        Args:
            path: Output file path (e.g., "videos/training.mp4")
            screen: Pygame surface to capture
            fps: Output video FPS
            sample_every: Capture every N-th frame (speeds up training videos)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        w, h = screen.get_size()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.path = path
        self.sample_every = sample_every
        self.frame_count = 0
        self.captured = 0

        if not self.writer.isOpened():
            print(f"  [WARN] Could not open video writer for {path}")

    def capture(self, screen: pygame.Surface):
        """Capture current frame. Call after pygame.display.flip()."""
        self.frame_count += 1
        if self.frame_count % self.sample_every != 0:
            return

        # Pygame surface → numpy array → BGR for OpenCV
        frame = pygame.surfarray.array3d(screen)  # (W, H, 3) RGB
        frame = frame.transpose(1, 0, 2)          # (H, W, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
        self.captured += 1

    def finish(self):
        """Finalize and close the video file."""
        self.writer.release()
        if self.captured > 0:
            print(f"  Video saved: {self.path} ({self.captured} frames)")
        else:
            print(f"  [WARN] No frames captured for {self.path}")
