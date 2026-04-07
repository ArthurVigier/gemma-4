"""
System 1 Pilot: Reactive flight controller and obstacle avoidance.
"""

from __future__ import annotations
import enum
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

from .ethics_guard import EthicsGuard
from .collision_avoidance import CollisionAvoidance

logger = logging.getLogger(__name__)

class PilotStatus(enum.Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    WAYPOINT_REACHED = "waypoint_reached"
    OBSTACLE_DETECTED = "obstacle_detected"
    OOD_ENCOUNTERED = "ood_encountered"
    FAILSAFE = "failsafe"

@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float | None = None
    description: str = ""

class System1Pilot:
    """Reactive Edge Pilot with Industrial MAVLink support."""
    
    def __init__(self, drone_id: str = "drone_0", adapter: Optional[Any] = None, mav_link: Optional[MAVLinkAdapter] = None):
        self.drone_id = drone_id
        self.adapter = adapter      # Isaac Sim Physics
        self.mav_link = mav_link    # PX4 Flight Stack
        self.current_status = PilotStatus.IDLE
        self.current_waypoint: Waypoint | None = None
        self.local_pose = (0.0, 0.0, 0.0)
        self.battery_percent = 100.0
        
        # Reactive Safety Layer
        self.avoidance = CollisionAvoidance()
        self.neighbors: List[Tuple[float, float, float]] = []

    async def initialize_flight(self):
        """Prepares the drone for a SAR mission."""
        if self.mav_link:
            await self.mav_link.connect()
            await self.mav_link.arm_and_takeoff()
            self.current_status = PilotStatus.NAVIGATING

    async def step(self, dt: float = 0.01) -> PilotStatus:
        """High-frequency control loop step (Asynchronous)."""
        
        # 1. Update Telemetry from real firmware
        if self.mav_link:
            telemetry = await self.mav_link.get_telemetry()
            # Convert NED for local logic (Simplified)
            self.local_pose = (0.0, 0.0, telemetry["alt"]) 

        if self.current_status != PilotStatus.NAVIGATING:
            return self.current_status
            
        # 2. Reactive Collision Avoidance
        if self.current_waypoint:
            target = (self.current_waypoint.x, self.current_waypoint.y, self.current_waypoint.z)
            safe_velocity = self.avoidance.filter_velocity(self.local_pose, target, self.neighbors)
            
            # 3. Send filtered command to PX4
            if self.mav_link:
                # Assuming simple NED movement for the demo
                await self.mav_link.goto_ned(safe_velocity[0], safe_velocity[1], -safe_velocity[2])

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
