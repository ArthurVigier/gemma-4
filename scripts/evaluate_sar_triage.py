"""
SAR Triage Evaluation Script.
Tests the Mastermind's ability to prioritize life-saving actions in crisis scenarios.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.system2_mastermind import System2Mastermind
from arc_drone.system1_pilot import System1Pilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def evaluate_scenario(scenario_path: Path):
    scenario = json.loads(scenario_path.read_text())
    logger.info("Evaluating Scenario: %s (%s)", scenario["scenario_id"], scenario["type"])
    
    mastermind = System2Mastermind(use_real_model=False) # Simulated for code logic check
    pilot = System1Pilot(drone_id="Rescue_Alpha")
    
    # 1. Initial State
    fleet_state = [pilot.get_semantic_state()]
    
    # 2. Simulate detection of multiple entities
    entities = scenario["entities"]
    logger.info("Drone detected %d entities of interest.", len(entities))
    
    # 3. Request triage from Mastermind
    # We pass all entities as active emergencies to see how it chooses
    emergencies = [
        {
            "drone_id": pilot.drone_id,
            "description": e["description"],
            "pose": e["pose"],
            "severity": "CRITICAL" if e["priority"] == 1 else "MEDIUM"
        } 
        for e in entities
    ]
    
    decision = mastermind.reason_sar(fleet_state, emergencies)
    
    logger.info("MASTERMIND DECISION:")
    logger.info("  Command: %s", decision["command"])
    logger.info("  Reasoning: %s", decision["reasoning"])
    
    # Basic validation: Did it choose a priority 1 victim?
    reasoning = decision["reasoning"].lower()
    if "life" in reasoning or "victim" in reasoning or "hiker" in reasoning:
        logger.info("RESULT: PASS (Prioritized life-saving action)")
    else:
        logger.warning("RESULT: FAIL (Did not prioritize correctly)")

def main():
    scenario_dir = Path("data/sar_scenarios")
    for s_file in scenario_dir.glob("*.json"):
        evaluate_scenario(s_file)
        print("-" * 50)

if __name__ == "__main__":
    main()
