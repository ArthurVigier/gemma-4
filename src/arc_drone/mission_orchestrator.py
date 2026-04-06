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
from .visual_crisis_analyzer import VisualCrisisAnalyzer

logger = logging.getLogger(__name__)

class MissionOrchestrator:
    def __init__(self, pilots: List[System1Pilot], mastermind: System2Mastermind):
        self.pilots = pilots
        self.mastermind = mastermind
        self.is_running = False
        self.active_emergencies: List[dict[str, Any]] = []
        
        # New: Visual analyzers for each pilot to maintain temporal context
        self.visual_analyzers = {p.drone_id: VisualCrisisAnalyzer() for p in pilots}

    def run_mission(self, duration_steps: int = 1000, dt: float = 0.01):
        """Main execution loop for SAR missions."""
        self.is_running = True
        logger.info("Starting SAR Mission Orchestrator with %d drones.", len(self.pilots))
        
        current_time = 0.0
        for step in range(duration_steps):
            current_time += dt
            swarm_state = [p.get_semantic_state() for p in self.pilots]

            for pilot in self.pilots:
                # 1. Pilot takes a physical step
                status = pilot.step(dt)
                
                # 2. Buffer visual data for temporal reasoning
                if pilot.adapter:
                    frame = pilot.adapter.get_camera_image()
                    self.visual_analyzers[pilot.drone_id].push_frame(frame, current_time)
                
                # 3. Handle emergencies
                if status == PilotStatus.OOD_ENCOUNTERED:
                    logger.warning("CRITICAL: OOD detected by %s. Requesting Mastermind Triage...", pilot.drone_id)
                    
                    # Get the visual clip for Gemma-4 26B MoE
                    clip = self.visual_analyzers[pilot.drone_id].get_crisis_clip()
                    
                    emergency = {
                        "drone_id": pilot.drone_id,
                        "description": "Unknown anomaly with temporal patterns.",
                        "pose": pilot.local_pose,
                        "severity": "HIGH"
                    }
                    self.active_emergencies.append(emergency)
                    
                    # Strategic reasoning with MULTI-FRAME input
                    strategy = self.mastermind.reason_sar(
                        fleet_state=swarm_state, 
                        emergencies=self.active_emergencies,
                        visual_input=clip
                    )
                    self.mastermind.execute_strategy(pilot, strategy)
                    
                    self.active_emergencies.clear()
                    self.visual_analyzers[pilot.drone_id].clear()
            
            if step % 200 == 0:
                logger.info("Step %d | SAR Status | Fleet: %s", step, [p.current_status.value for p in self.pilots])
            
            time.sleep(dt)

        self.is_running = False
        logger.info("Mission concluded.")
