"""
MAVSDK Adapter: The industrial bridge to PX4 flight firmware.

Handles the MAVLink protocol to send high-level commands (waypoints, arming, 
takeoff) to real or simulated drones.
"""

from __future__ import annotations
import asyncio
import logging
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
from mavsdk.telemetry import FlightMode

logger = logging.getLogger(__name__)

class MAVLinkAdapter:
    """Industrial-grade flight controller interface."""
    
    def __init__(self, system_address: str = "udp://:14540"):
        self.system_address = system_address
        self.drone = System()
        self.is_connected = False

    async def connect(self):
        """Initializes connection to the drone."""
        logger.info("Connecting to drone at %s...", self.system_address)
        await self.drone.connect(system_address=self.system_address)
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                logger.info("MAVLink: Connection successful!")
                self.is_connected = True
                break
        return True

    async def arm_and_takeoff(self, altitude: float = 2.0):
        """Standard safety procedure for SAR drones."""
        logger.info("MAVLink: Arming motors...")
        try:
            await self.drone.action.arm()
            logger.info("MAVLink: Taking off to %.1fm...", altitude)
            await self.drone.action.set_takeoff_altitude(altitude)
            await self.drone.action.takeoff()
        except Exception as e:
            logger.error("MAVLink: Launch failed: %s", e)

    async def goto_ned(self, north: float, east: float, down: float, yaw: float = 0.0):
        """Sends the drone to a specific North-East-Down coordinate (Offboard)."""
        logger.info("MAVLink: Strategic Waypoint -> N:%.1f E:%.1f D:%.1f", north, east, down)
        
        # Offboard mode is required for reactive real-time control
        try:
            # We must send a setpoint BEFORE starting offboard mode
            await self.drone.offboard.set_position_ned(PositionNedYaw(north, east, down, yaw))
            await self.drone.offboard.start()
        except OffboardError as e:
            logger.error("MAVLink: Offboard failed: %s", e)

    async def get_telemetry(self):
        """Streams real-time health data for the Mastermind."""
        # This would be an async generator in a real deployment
        async for position in self.drone.telemetry.position():
            return {
                "lat": position.latitude_deg,
                "lon": position.longitude_deg,
                "alt": position.relative_altitude_m
            }
            break # Get only one for polling
