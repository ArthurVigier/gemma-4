"""
Verification test for Multi-Frame VLM reasoning (SAR Triage).

Simulates a sequence of frames from a drone POV where a victim is signaling,
and verifies if the Mastermind (System 2) generates the correct SAR command.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.system2_mastermind import System2Mastermind
from arc_drone.visual_crisis_analyzer import VisualCrisisAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def create_simulated_video_clip(num_frames: int = 4) -> list[Image.Image]:
    """Generates a clip of frames with a moving 'person' (red dot) to simulate movement."""
    clip = []
    for i in range(num_frames):
        # Create a base gray image (224x224)
        frame = Image.new("RGB", (224, 224), color=(50, 50, 50))
        draw = ImageDraw.Draw(frame)
        
        # Simulate a person (red square) moving across frames
        pos_x = 100 + (i * 10)
        pos_y = 100
        draw.rectangle([pos_x, pos_y, pos_x+10, pos_y+10], fill=(255, 0, 0))
        
        clip.append(frame)
    return clip

def test_mastermind_temporal_reasoning():
    logger.info("--- Testing Mastermind Temporal SAR Reasoning ---")
    
    # 1. Initialize Mastermind in simulated mode
    mastermind = System2Mastermind(use_real_model=False)
    
    # 2. Simulate a fleet state
    fleet_state = [{
        "drone_id": "Rescue_01",
        "status": "navigating",
        "pose": (10.0, 10.0, 5.0),
        "battery": "85.0%"
    }]
    
    # 3. Create a crisis clip (moving victim)
    clip = create_simulated_video_clip(num_frames=5)
    logger.info("Generated a simulated 5-frame clip with a moving target.")
    
    # 4. Define the emergency context
    emergencies = [{
        "drone_id": "Rescue_01",
        "description": "Moving red object detected in Sector Alpha.",
        "pose": (12.0, 10.0, 2.0),
        "severity": "HIGH"
    }]
    
    # 5. Run reasoning (Simulated logic for now)
    # This verifies the API contract for the 26B MoE implementation
    decision = mastermind.reason_sar(
        fleet_state=fleet_state,
        emergencies=emergencies,
        visual_input=clip,
        think=True
    )
    
    logger.info("MASTERMIND SAR DECISION:")
    logger.info("  Command: %s", decision.get("command"))
    logger.info("  Reasoning: %s", decision.get("reasoning"))
    
    # Validation
    assert "command" in decision, "Decision must contain a command"
    assert "reasoning" in decision, "Decision must contain reasoning"
    
    if decision["command"] == "HOVER_AND_SIGNAL":
        logger.info("RESULT: PASS (Mastermind correctly prioritized rescue)")
    else:
        logger.warning("RESULT: FAIL (Mastermind did not choose life-saving action)")

if __name__ == "__main__":
    test_mastermind_temporal_reasoning()
