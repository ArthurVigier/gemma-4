"""
System 1 Pilot: Reactive flight controller and obstacle avoidance.

This module acts as the interface for high-frequency flight control (>50Hz).
It accepts high-level semantic waypoints from the System 2 Mastermind and
executes them using local reactive policies (e.g., RL-based avoidance).
"""

from __future__ import annotations
import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

class PilotStatus(enum.Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    WAYPOINT_REACHED = "waypoint_reached"
    OBSTACLE_DETECTED = "obstacle_detected"
    OOD_ENCOUNTERED = "ood_encountered"  # Out-Of-Distribution (unknown object)
    FAILSAFE = "failsafe"

@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float | None = None
    description: str = ""

class System1Pilot:
    """Reactive Edge Pilot."""
    
    def __init__(self, drone_id: str = "drone_0", adapter: Optional[Any] = None):
        self.drone_id = drone_id
        self.adapter = adapter  # IsaacPX4Adapter
        self.current_status = PilotStatus.IDLE
        self.current_waypoint: Waypoint | None = None
        self.local_pose = (0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0, 0.0)
        self.battery_percent = 100.0
        self.sensor_health = "OK"
        
    def set_waypoint(self, waypoint: Waypoint) -> None:
        """Accept a new strategic target from the Mastermind."""
        self.current_waypoint = waypoint
        self.current_status = PilotStatus.NAVIGATING
        logger.info("[%s] New strategic waypoint received: %s", self.drone_id, waypoint.description)

    def step(self, dt: float = 0.01) -> PilotStatus:
        """High-frequency control loop step."""
        if self.adapter:
            # Sync pose with high-fidelity simulator physics
            self.local_pose = self.adapter.get_ground_truth_pose()
            
        if self.current_status != PilotStatus.NAVIGATING:
            return self.current_status
            
        self.battery_percent -= 0.005
        
        # OOD Logic (Could be triggered by real sensor anomaly in Isaac)
        import random
        if random.random() < 0.001:
            self.current_status = PilotStatus.OOD_ENCOUNTERED
            logger.warning("[%s] OOD Encountered! Mastermind help required.", self.drone_id)
            
        return self.current_status

    def get_semantic_state(self) -> dict[str, Any]:
        """Returns a compressed semantic summary for the Mastermind."""
        return {
            "drone_id": self.drone_id,
            "status": self.current_status.value,
            "pose": self.local_pose,
            "velocity": self.velocity,
            "battery": f"{self.battery_percent:.1f}%",
            "sensors": self.sensor_health,
            "target": self.current_waypoint.description if self.current_waypoint else None
        }
