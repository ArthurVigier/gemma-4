"""
Mission Orchestrator: Coordinates System 1 (Pilot) and System 2 (Mastermind).
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import List, Any
from .system1_pilot import System1Pilot, PilotStatus
from .system2_mastermind import System2Mastermind
from .visual_crisis_analyzer import VisualCrisisAnalyzer
from .sar_comm_hub import SARCommunicationHub

logger = logging.getLogger(__name__)

class MissionOrchestrator:
    def __init__(self, pilots: List[System1Pilot], mastermind: System2Mastermind):
        self.pilots = pilots
        self.mastermind = mastermind
        self.is_running = False
        self.active_emergencies: List[dict[str, Any]] = []
        
        self.visual_analyzers = {p.drone_id: VisualCrisisAnalyzer() for p in pilots}
        self.comm_hub = SARCommunicationHub()

    async def run_mission(self, duration_steps: int = 1000, dt: float = 0.01):
        """Main asynchronous execution loop for SAR missions."""
        self.is_running = True
        logger.info("Starting SAR Mission Orchestrator (Asynchronous) with %d drones.", len(self.pilots))
        
        # 1. Initialize MAVLink flight for all drones
        for pilot in self.pilots:
            await pilot.initialize_flight()

        current_time = 0.0
        for step in range(duration_steps):
            current_time += dt
            swarm_state = [p.get_semantic_state() for p in self.pilots]

            # 2. Local V2V Sync
            for i, pilot in enumerate(self.pilots):
                others = [p.local_pose for j, p in enumerate(self.pilots) if i != j]
                pilot.update_neighbors(others)

            # 3. High-Frequency Step
            for pilot in self.pilots:
                status = await pilot.step(dt)
                
                # Visual Capture
                if pilot.adapter:
                    frame = pilot.adapter.get_camera_image()
                    self.visual_analyzers[pilot.drone_id].push_frame(frame, current_time)
                
                # 4. Crisis Intervention (Mastermind)
                if status == PilotStatus.OOD_ENCOUNTERED:
                    logger.warning("EVENT: Critical anomaly detected by %s. Requesting strategic triage...", pilot.drone_id)
                    clip = self.visual_analyzers[pilot.drone_id].get_crisis_clip()
                    
                    emergency = {
                        "drone_id": pilot.drone_id,
                        "description": "Moving human silhouette on flood-damaged rooftop.",
                        "pose": pilot.local_pose,
                        "severity": "HIGH"
                    }
                    self.active_emergencies.append(emergency)
                    
                    # Mastermind reasoning (System 2)
                    strategy = self.mastermind.reason_sar(swarm_state, self.active_emergencies, visual_input=clip)
                    self.comm_hub.broadcast_sitrep(pilot.drone_id, strategy)
                    self.mastermind.execute_strategy(pilot, strategy)
                    
                    self.active_emergencies.clear()
            
            if step % 200 == 0:
                logger.info("Step %d | Fleet Pose: %s", step, [p.local_pose for p in self.pilots])
            
            await asyncio.sleep(dt)

        self.is_running = False
        logger.info("Mission concluded.")
