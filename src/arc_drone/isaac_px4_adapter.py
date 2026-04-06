"""
Isaac Sim + Pegasus Adapter for ARC-Drone-Bench.

Handles the high-fidelity physics environment and MAVLink connection to PX4.
Requires NVIDIA Isaac Sim 2023.1.1+ and the Pegasus Simulator extension.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

# These imports are only available inside the Isaac Sim environment
try:
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

logger = logging.getLogger(__name__)

class IsaacPX4Adapter:
    """Interface between the logical Pilot and the physical Isaac Sim world."""
    
    def __init__(self, px4_dir: str = "~/PX4-Autopilot", vehicle_model: str = "iris"):
        if not ISAAC_AVAILABLE:
            logger.error("Isaac Sim / Pegasus modules not found. Ensure you are running with 'isaac_run'.")
            return

        self.px4_dir = px4_dir
        self.vehicle_model = vehicle_model
        
        # 1. Initialize Pegasus Interface
        self.pg = PegasusInterface()
        self.vehicle: Optional[Multirotor] = None
        
    def spawn_drone(self, drone_id: str, init_pos: list[float] = [0.0, 0.0, 0.5]):
        """Spawns a drone in Isaac Sim and connects it to a new PX4 SITL instance."""
        logger.info("[%s] Spawning high-fidelity drone in Isaac Sim...", drone_id)
        
        # Configure MAVLink backend for PX4
        # Note: In a real multi-drone setup, baseport must increment per drone
        mav_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "connection_type": "tcpin",
            "connection_baseport": 4560,
            "px4_autolaunch": True,
            "px4_dir": self.px4_dir,
            "px4_vehicle_model": self.vehicle_model
        })
        
        config = MultirotorConfig()
        config.backends = [PX4MavlinkBackend(mav_config)]
        
        # Spawn in the USD Stage
        self.vehicle = Multirotor(
            f"/World/{drone_id}",
            self.vehicle_model,
            config=config,
            init_pos=init_pos
        )
        return self.vehicle

    def update(self):
        """Advances the simulation by one step."""
        if ISAAC_AVAILABLE:
            self.pg.run() # This maintains the internal lockstep loop

    def get_ground_truth_pose(self) -> tuple[float, float, float]:
        """Returns the absolute position from Isaac physics (bypassing IMU noise)."""
        if self.vehicle:
            pos = self.vehicle.get_world_pose()[0] # returns ([x,y,z], [quat])
            return (pos[0], pos[1], pos[2])
        return (0.0, 0.0, 0.0)

    def get_camera_image(self) -> Image.Image:
        """Captures the photorealistic RGB frame from the drone's POV."""
        if self.vehicle:
            # TODO: Link to real Isaac Sim viewport capture
            pass
            
        # Return dummy frame for dev
        return Image.new("RGB", (224, 224), color=(30, 30, 30))
