"""
Visual Crisis Analyzer: Handles temporal visual data extraction for the Mastermind.

Collects and prepares video sequences (clips) from the drone's POV to be 
analyzed by Gemma-4 26B MoE for strategic SAR triage.
"""

from __future__ import annotations
import collections
import logging
from typing import List, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class VisualCrisisAnalyzer:
    """Extracts and prepares visual clips from drone sensors."""
    
    def __init__(self, max_frames: int = 10, sampling_frequency: float = 5.0):
        """
        Args:
            max_frames: Number of frames to keep in the rolling buffer.
            sampling_frequency: Hz at which frames should be captured.
        """
        self.max_frames = max_frames
        self.sampling_frequency = sampling_frequency
        
        # Rolling buffer for temporal context (video-like input)
        self.frame_buffer = collections.deque(maxlen=max_frames)
        self.last_capture_time = 0.0

    def push_frame(self, frame: Any, current_time: float):
        """Adds a new frame to the buffer based on the sampling frequency."""
        if (current_time - self.last_capture_time) >= (1.0 / self.sampling_frequency):
            # In Isaac Sim, frame might be a tensor or numpy array. 
            # We ensure it's a PIL Image for the VLM processor.
            if not isinstance(frame, Image.Image):
                # Simulated conversion logic
                try:
                    import numpy as np
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                except ImportError:
                    pass
            
            self.frame_buffer.append(frame)
            self.last_capture_time = current_time

    def get_crisis_clip(self) -> List[Image.Image]:
        """Returns the full temporal sequence for VLM reasoning."""
        if len(self.frame_buffer) == 0:
            logger.warning("Visual buffer is empty. Providing fallback blank frame.")
            return [Image.new("RGB", (224, 224), color=(0, 0, 0))]
            
        return list(self.frame_buffer)

    def clear(self):
        """Wipes the buffer after a mission event is resolved."""
        self.frame_buffer.clear()

    def summarize_visual_anomalies(self, clip: List[Image.Image]) -> str:
        """
        Optional: Pre-process clip using a smaller local vision model (System 1)
        to provide a textual hint to the Mastermind.
        """
        # Placeholder for a local object detector (e.g. YOLOv10-tiny)
        return "Multiple movement patterns detected in the center of the frames."
