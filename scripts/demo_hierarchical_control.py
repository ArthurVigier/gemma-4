"""
Multi-Drone Hierarchical Orchestration Demo.
"""

import asyncio
import logging
from src.arc_drone.system1_pilot import System1Pilot
from src.arc_drone.system2_mastermind import System2Mastermind
from src.arc_drone.mission_orchestrator import MissionOrchestrator
from src.arc_drone.mavlink_adapter import MAVLinkAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

async def main():
    # 1. Initialize MAVLink adapters (Simulation ports)
    # Drone Alpha on 14540, Bravo on 14541
    mav_a = MAVLinkAdapter(system_address="udp://:14540")
    
    # 2. Initialize Pilots with MAVLink
    pilot_a = System1Pilot(drone_id="Alpha", mav_link=mav_a)
    
    # 3. Mastermind
    mastermind = System2Mastermind(use_real_model=False)
    
    # 4. Orchestrator
    orchestrator = MissionOrchestrator([pilot_a], mastermind)
    
    # 5. Run Async Mission
    await orchestrator.run_mission(duration_steps=400)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
