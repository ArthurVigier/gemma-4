"""
Full Mission Demo: Hierarchical Mastermind + Pilot + Isaac Sim/Pegasus.
"""

from __future__ import annotations
import logging
import time
from src.arc_drone.system1_pilot import System1Pilot
from src.arc_drone.system2_mastermind import System2Mastermind
from src.arc_drone.mission_orchestrator import MissionOrchestrator
from src.arc_drone.isaac_px4_adapter import IsaacPX4Adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

def main():
    # 1. Initialize Isaac Sim Adapter (Requires running in Isaac Sim environment)
    # adapter = IsaacPX4Adapter(px4_dir="/workspace/PX4-Autopilot", vehicle_model="iris")
    # adapter.spawn_drone("Alpha")
    adapter = None # Mock for local dev without Isaac
    
    # 2. Initialize Hierarchical Layers
    pilot = System1Pilot(drone_id="Alpha", adapter=adapter)
    mastermind = System2Mastermind(use_real_model=False)
    
    # 3. Setup Orchestrator
    orchestrator = MissionOrchestrator([pilot], mastermind)
    
    # 4. Run Mission
    # The loop should advance both the orchestrator and the physics
    logger.info("Starting Mission in photorealistic simulation...")
    orchestrator.run_mission(duration_steps=100)

if __name__ == "__main__":
    main()
