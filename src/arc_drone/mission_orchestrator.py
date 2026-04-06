"""
Mission Orchestrator: Coordinates System 1 (Pilot) and System 2 (Mastermind).
Handles the frequency mismatch and triggers strategic reasoning when needed.
"""

from __future__ import annotations
import logging
import time
from typing import List
from .system1_pilot import System1Pilot, PilotStatus
from .system2_mastermind import System2Mastermind

logger = logging.getLogger(__name__)

class MissionOrchestrator:
    def __init__(self, pilots: List[System1Pilot], mastermind: System2Mastermind):
        self.pilots = pilots
        self.mastermind = mastermind
        self.is_running = False

    def run_mission(self, duration_steps: int = 1000, dt: float = 0.01):
        """Main execution loop."""
        self.is_running = True
        logger.info("Starting Mission Orchestrator with %d drones.", len(self.pilots))
        
        # Initial strategic planning
        swarm_state = [p.get_semantic_state() for p in self.pilots]
        strategy = self.mastermind.reason(swarm_state)
        self.mastermind.execute_strategy(self.pilots[0], strategy)

        for step in range(duration_steps):
            for pilot in self.pilots:
                status = pilot.step(dt)
                
                # Check for events requiring Mastermind intervention
                if status == PilotStatus.OOD_ENCOUNTERED:
                    logger.warning("EVENT: OOD detected by %s. Triggering Mastermind reasoning...", pilot.drone_id)
                    
                    ood_info = pilot.get_semantic_state()
                    ood_info["observation"] = "Unknown dynamic obstacle detected in flight path."
                    
                    # High-latency reasoning step (System 2)
                    intervention = self.mastermind.reason(
                        swarm_state=[p.get_semantic_state() for p in self.pilots],
                        ood_event=ood_info
                    )
                    
                    self.mastermind.execute_strategy(pilot, intervention)
                
                elif pilot.battery_percent < 15.0:
                    logger.warning("EVENT: Low battery on %s. Mastermind calculating RTH...", pilot.drone_id)
                    # Mastermind would handle Return To Home logic here
            
            if step % 200 == 0:
                logger.info("Step %d | Fleet Status: %s", step, [p.current_status.value for p in self.pilots])
            
            time.sleep(dt)

        self.is_running = False
        logger.info("Mission concluded.")
