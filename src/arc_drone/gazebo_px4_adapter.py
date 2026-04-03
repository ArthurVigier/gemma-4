"""Gazebo Harmonic + PX4 SITL + ROS 2 adapter.

This adapter follows the current PX4 ROS 2 offboard-control guidance:

- image observations arrive through ROS 2 image topics, typically bridged from Gazebo
- offboard control is driven by pairing `OffboardControlMode` and `TrajectorySetpoint`
- arming and mode switching are sent with `VehicleCommand`
- PX4 expects world-frame setpoints in NED, while ROS commonly uses ENU

The adapter is intentionally split in two layers:

1. a pure-Python layer that is fully unit-testable without ROS 2 installed
2. an optional ROS 2 binding that creates publishers/subscriptions when
   `sensor_msgs` and `px4_msgs` are available in the runtime environment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import nan
from typing import Any, Callable, Protocol

import numpy as np

from .arc_types import DroneAction
from .simulators import DroneSimulator, SimObservation

try:  # pragma: no cover - optional runtime dependency
    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleCommandAck,
        VehicleOdometry,
        VehicleStatus,
    )
    from sensor_msgs.msg import Image
except ImportError:  # pragma: no cover - exercised through dictionary fallback in tests
    Image = None
    OffboardControlMode = None
    TrajectorySetpoint = None
    VehicleCommand = None
    VehicleCommandAck = None
    VehicleOdometry = None
    VehicleStatus = None


@dataclass(slots=True)
class GazeboPx4Topics:
    """ROS 2 topics used by the Gazebo/PX4 adapter."""

    image_topic: str = "/camera/image_raw"
    offboard_control_mode_topic: str = "/fmu/in/offboard_control_mode"
    trajectory_setpoint_topic: str = "/fmu/in/trajectory_setpoint"
    vehicle_command_topic: str = "/fmu/in/vehicle_command"
    vehicle_status_topic: str = "/fmu/out/vehicle_status"
    vehicle_odometry_topic: str = "/fmu/out/vehicle_odometry"
    vehicle_command_ack_topic: str = "/fmu/out/vehicle_command_ack"


@dataclass(slots=True)
class GazeboPx4AdapterConfig:
    """Adapter settings for the default Gazebo Harmonic + PX4 SITL stack."""

    simulator_name: str = "gazebo_harmonic_px4_sitl"
    image_encoding: str = "rgb8"
    qos_depth: int = 10
    offboard_warmup_cycles: int = 10
    transition_timeout_s: float = 3.0
    require_preflight_checks: bool = True
    validate_px4_state: bool = True
    target_system: int = 1
    target_component: int = 1
    source_system: int = 1
    source_component: int = 1
    default_yaw_rad: float = 0.0
    topics: GazeboPx4Topics = field(default_factory=GazeboPx4Topics)


@dataclass(slots=True)
class Px4VehicleStatusTelemetry:
    """Relevant commander state from PX4 `VehicleStatus`."""

    timestamp_us: int
    nav_state: int
    arming_state: int
    pre_flight_checks_pass: bool
    failsafe: bool


@dataclass(slots=True)
class Px4OdometryTelemetry:
    """Relevant PX4 odometry state."""

    timestamp_us: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    pose_frame: int
    velocity_frame: int
    quality: int


@dataclass(slots=True)
class Px4CommandAckTelemetry:
    """Latest acknowledgement for a PX4 vehicle command."""

    timestamp_us: int
    command: int
    result: int
    result_param1: int
    result_param2: int


@dataclass(slots=True)
class Px4ControlState:
    """Aggregated PX4 control readiness snapshot."""

    telemetry_available: bool
    preflight_checks_pass: bool
    preflight_checks_known: bool
    armed: bool
    offboard_enabled: bool
    failsafe_active: bool
    last_command_error: str | None
    ready_for_control: bool


class PublisherProtocol(Protocol):
    """Small publish protocol to support ROS publishers and test doubles."""

    def publish(self, message: Any) -> None:
        """Publishes one message."""


@dataclass(slots=True)
class GazeboPx4Publishers:
    """Publisher handles used by the adapter."""

    offboard_control_mode: PublisherProtocol | None = None
    trajectory_setpoint: PublisherProtocol | None = None
    vehicle_command: PublisherProtocol | None = None


class Px4MessageFactory(Protocol):
    """Message factory used to decouple tests from ROS message classes."""

    def offboard_control_mode(self, *, timestamp_us: int, velocity: bool) -> Any:
        """Builds an OffboardControlMode message."""

    def trajectory_setpoint(
        self,
        *,
        timestamp_us: int,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        yaw: float,
        yaw_rate: float,
    ) -> Any:
        """Builds a TrajectorySetpoint message."""

    def vehicle_command(
        self,
        *,
        timestamp_us: int,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        target_system: int = 1,
        target_component: int = 1,
        source_system: int = 1,
        source_component: int = 1,
    ) -> Any:
        """Builds a VehicleCommand message."""


class DictionaryMessageFactory:
    """Test-friendly factory that emits plain dictionaries."""

    VEHICLE_CMD_DO_SET_MODE = 176
    VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
    VEHICLE_CMD_RESULT_ACCEPTED = 0
    VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED = 1
    VEHICLE_CMD_RESULT_DENIED = 2
    VEHICLE_CMD_RESULT_UNSUPPORTED = 3
    VEHICLE_CMD_RESULT_FAILED = 4
    VEHICLE_CMD_RESULT_IN_PROGRESS = 5
    VEHICLE_CMD_RESULT_CANCELLED = 6
    VEHICLE_CMD_RESULT_COMMAND_LONG_ONLY = 7
    VEHICLE_CMD_RESULT_COMMAND_INT_ONLY = 8
    VEHICLE_CMD_RESULT_UNSUPPORTED_MAV_FRAME = 9
    ARMING_STATE_DISARMED = 1
    ARMING_STATE_ARMED = 2
    NAVIGATION_STATE_OFFBOARD = 14

    def offboard_control_mode(self, *, timestamp_us: int, velocity: bool) -> dict[str, Any]:
        return {
            "timestamp": timestamp_us,
            "position": False,
            "velocity": velocity,
            "acceleration": False,
            "attitude": False,
            "body_rate": False,
            "thrust_and_torque": False,
            "direct_actuator": False,
        }

    def trajectory_setpoint(
        self,
        *,
        timestamp_us: int,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        yaw: float,
        yaw_rate: float,
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp_us,
            "position": list(position),
            "velocity": list(velocity),
            "acceleration": [nan, nan, nan],
            "jerk": [nan, nan, nan],
            "yaw": yaw,
            "yawspeed": yaw_rate,
        }

    def vehicle_command(
        self,
        *,
        timestamp_us: int,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        target_system: int = 1,
        target_component: int = 1,
        source_system: int = 1,
        source_component: int = 1,
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp_us,
            "command": command,
            "param1": param1,
            "param2": param2,
            "param3": 0.0,
            "param4": 0.0,
            "param5": 0.0,
            "param6": 0.0,
            "param7": 0.0,
            "target_system": target_system,
            "target_component": target_component,
            "source_system": source_system,
            "source_component": source_component,
            "confirmation": 0,
            "from_external": True,
        }


class Ros2Px4MessageFactory(DictionaryMessageFactory):
    """PX4 ROS 2 message factory using `px4_msgs` when available."""

    def __init__(self) -> None:
        if (
            OffboardControlMode is None
            or TrajectorySetpoint is None
            or VehicleCommand is None
            or VehicleStatus is None
            or VehicleOdometry is None
            or VehicleCommandAck is None
        ):
            raise RuntimeError("px4_msgs is not available in this environment.")

    def offboard_control_mode(self, *, timestamp_us: int, velocity: bool) -> OffboardControlMode:
        msg = OffboardControlMode()
        msg.timestamp = timestamp_us
        msg.position = False
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        return msg

    def trajectory_setpoint(
        self,
        *,
        timestamp_us: int,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        yaw: float,
        yaw_rate: float,
    ) -> TrajectorySetpoint:
        msg = TrajectorySetpoint()
        msg.timestamp = timestamp_us
        msg.position = list(position)
        msg.velocity = list(velocity)
        msg.acceleration = [nan, nan, nan]
        msg.jerk = [nan, nan, nan]
        msg.yaw = yaw
        msg.yawspeed = yaw_rate
        return msg

    def vehicle_command(
        self,
        *,
        timestamp_us: int,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        target_system: int = 1,
        target_component: int = 1,
        source_system: int = 1,
        source_component: int = 1,
    ) -> VehicleCommand:
        msg = VehicleCommand()
        msg.timestamp = timestamp_us
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
        msg.command = int(command)
        msg.target_system = int(target_system)
        msg.target_component = int(target_component)
        msg.source_system = int(source_system)
        msg.source_component = int(source_component)
        msg.confirmation = 0
        msg.from_external = True
        return msg


@dataclass(slots=True)
class GazeboPx4CommandBundle:
    """All PX4 messages needed for one inference cycle."""

    offboard_control_mode: Any
    trajectory_setpoint: Any
    vehicle_commands: list[Any]


class GazeboPx4Adapter(DroneSimulator):
    """Adapter between ROS/Gazebo observations and PX4 offboard control topics."""

    def __init__(
        self,
        config: GazeboPx4AdapterConfig | None = None,
        message_factory: Px4MessageFactory | None = None,
        publishers: GazeboPx4Publishers | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self.config = config or GazeboPx4AdapterConfig()
        self.message_factory = message_factory or DictionaryMessageFactory()
        self.publishers = publishers or GazeboPx4Publishers()
        self._clock_ns = clock_ns or (lambda: 0)
        self._latest_observation: SimObservation | None = None
        self._latest_vehicle_status: Px4VehicleStatusTelemetry | None = None
        self._latest_odometry: Px4OdometryTelemetry | None = None
        self._latest_command_acks: dict[int, Px4CommandAckTelemetry] = {}
        self._offboard_setpoint_counter = 0
        self._offboard_enabled = False
        self._transition_started_us: int | None = None
        self._subscriptions: list[Any] = []

    def bind_ros_interfaces(self, ros_node: Any) -> None:
        """Creates ROS publishers/subscriptions on a live ROS 2 node."""

        if Image is None:
            raise RuntimeError("sensor_msgs is not available in this environment.")
        if not isinstance(self.message_factory, Ros2Px4MessageFactory):
            self.message_factory = Ros2Px4MessageFactory()

        self.publishers = GazeboPx4Publishers(
            offboard_control_mode=ros_node.create_publisher(
                OffboardControlMode,
                self.config.topics.offboard_control_mode_topic,
                self.config.qos_depth,
            ),
            trajectory_setpoint=ros_node.create_publisher(
                TrajectorySetpoint,
                self.config.topics.trajectory_setpoint_topic,
                self.config.qos_depth,
            ),
            vehicle_command=ros_node.create_publisher(
                VehicleCommand,
                self.config.topics.vehicle_command_topic,
                self.config.qos_depth,
            ),
        )
        image_subscription = ros_node.create_subscription(
            Image,
            self.config.topics.image_topic,
            self.ingest_image_message,
            self.config.qos_depth,
        )
        status_subscription = ros_node.create_subscription(
            VehicleStatus,
            self.config.topics.vehicle_status_topic,
            self.ingest_vehicle_status_message,
            self.config.qos_depth,
        )
        odometry_subscription = ros_node.create_subscription(
            VehicleOdometry,
            self.config.topics.vehicle_odometry_topic,
            self.ingest_vehicle_odometry_message,
            self.config.qos_depth,
        )
        ack_subscription = ros_node.create_subscription(
            VehicleCommandAck,
            self.config.topics.vehicle_command_ack_topic,
            self.ingest_vehicle_command_ack_message,
            self.config.qos_depth,
        )
        self._subscriptions = [image_subscription, status_subscription, odometry_subscription, ack_subscription]

    def ingest_image_message(self, image_message: Any) -> SimObservation:
        """Decodes a ROS image message and stores it as the latest observation."""

        rgb = self.decode_image(image_message)
        observation = SimObservation(
            rgb_image=rgb,
            timestamp_ns=self.extract_timestamp_ns(image_message),
            simulator_name=self.config.simulator_name,
        )
        self._latest_observation = observation
        return observation

    def get_observation(self) -> SimObservation:
        """Returns the latest simulator observation received over ROS 2."""

        if self._latest_observation is None:
            raise RuntimeError("No Gazebo/PX4 observation has been received yet.")
        return self._latest_observation

    def latest_vehicle_status(self) -> Px4VehicleStatusTelemetry | None:
        """Returns the latest decoded `VehicleStatus` telemetry."""

        return self._latest_vehicle_status

    def latest_odometry(self) -> Px4OdometryTelemetry | None:
        """Returns the latest decoded `VehicleOdometry` telemetry."""

        return self._latest_odometry

    def control_state(self) -> Px4ControlState:
        """Builds the current PX4 control readiness state."""

        status = self._latest_vehicle_status
        command_error = self._latest_command_error()
        telemetry_available = status is not None
        preflight_known = status is not None
        preflight_pass = bool(status.pre_flight_checks_pass) if status is not None else False
        armed = bool(status and status.arming_state == DictionaryMessageFactory.ARMING_STATE_ARMED)
        offboard = bool(status and status.nav_state == DictionaryMessageFactory.NAVIGATION_STATE_OFFBOARD)
        failsafe = bool(status.failsafe) if status is not None else False
        ready = telemetry_available and preflight_pass and armed and offboard and not failsafe and command_error is None
        return Px4ControlState(
            telemetry_available=telemetry_available,
            preflight_checks_pass=preflight_pass,
            preflight_checks_known=preflight_known,
            armed=armed,
            offboard_enabled=offboard,
            failsafe_active=failsafe,
            last_command_error=command_error,
            ready_for_control=ready,
        )

    def send_action(self, action: DroneAction) -> None:
        """Publishes the PX4 offboard heartbeat and the corresponding setpoint."""

        bundle = self.build_command_bundle(action)
        self._publish(self.publishers.offboard_control_mode, bundle.offboard_control_mode)
        self._publish(self.publishers.trajectory_setpoint, bundle.trajectory_setpoint)
        for command in bundle.vehicle_commands:
            self._publish(self.publishers.vehicle_command, command)
        self.validate_px4_control_state()

    def build_command_bundle(self, action: DroneAction) -> GazeboPx4CommandBundle:
        """Builds all PX4 commands needed for one policy output."""

        timestamp_us = self.now_us()
        offboard = self.message_factory.offboard_control_mode(timestamp_us=timestamp_us, velocity=True)
        trajectory = self.message_factory.trajectory_setpoint(
            timestamp_us=timestamp_us,
            position=(nan, nan, nan),
            velocity=self.ros_enu_to_px4_ned(action.velocity_xyz),
            yaw=self.config.default_yaw_rad,
            yaw_rate=-float(action.yaw_rate),
        )

        self._offboard_setpoint_counter += 1
        commands: list[Any] = []
        if not self._offboard_enabled and self._offboard_setpoint_counter >= self.config.offboard_warmup_cycles:
            if self.config.require_preflight_checks and self._latest_vehicle_status is not None:
                if not self._latest_vehicle_status.pre_flight_checks_pass:
                    return GazeboPx4CommandBundle(
                        offboard_control_mode=offboard,
                        trajectory_setpoint=trajectory,
                        vehicle_commands=[],
                    )
            commands.append(
                self.message_factory.vehicle_command(
                    timestamp_us=timestamp_us,
                    command=DictionaryMessageFactory.VEHICLE_CMD_DO_SET_MODE,
                    param1=1.0,
                    param2=6.0,
                    target_system=self.config.target_system,
                    target_component=self.config.target_component,
                    source_system=self.config.source_system,
                    source_component=self.config.source_component,
                )
            )
            commands.append(
                self.message_factory.vehicle_command(
                    timestamp_us=timestamp_us,
                    command=DictionaryMessageFactory.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0,
                    target_system=self.config.target_system,
                    target_component=self.config.target_component,
                    source_system=self.config.source_system,
                    source_component=self.config.source_component,
                )
            )
            self._offboard_enabled = True
            self._transition_started_us = timestamp_us

        return GazeboPx4CommandBundle(
            offboard_control_mode=offboard,
            trajectory_setpoint=trajectory,
            vehicle_commands=commands,
        )

    def ingest_vehicle_status_message(self, status_message: Any) -> Px4VehicleStatusTelemetry:
        """Stores the latest PX4 `VehicleStatus` telemetry."""

        telemetry = Px4VehicleStatusTelemetry(
            timestamp_us=int(getattr(status_message, "timestamp")),
            nav_state=int(getattr(status_message, "nav_state")),
            arming_state=int(getattr(status_message, "arming_state")),
            pre_flight_checks_pass=bool(getattr(status_message, "pre_flight_checks_pass", False)),
            failsafe=bool(getattr(status_message, "failsafe", False)),
        )
        self._latest_vehicle_status = telemetry
        return telemetry

    def ingest_vehicle_odometry_message(self, odometry_message: Any) -> Px4OdometryTelemetry:
        """Stores the latest PX4 `VehicleOdometry` telemetry."""

        telemetry = Px4OdometryTelemetry(
            timestamp_us=int(getattr(odometry_message, "timestamp")),
            position=tuple(float(value) for value in getattr(odometry_message, "position")),
            velocity=tuple(float(value) for value in getattr(odometry_message, "velocity")),
            angular_velocity=tuple(float(value) for value in getattr(odometry_message, "angular_velocity")),
            pose_frame=int(getattr(odometry_message, "pose_frame")),
            velocity_frame=int(getattr(odometry_message, "velocity_frame")),
            quality=int(getattr(odometry_message, "quality", 0)),
        )
        self._latest_odometry = telemetry
        return telemetry

    def ingest_vehicle_command_ack_message(self, ack_message: Any) -> Px4CommandAckTelemetry:
        """Stores the latest PX4 `VehicleCommandAck` message by command id."""

        telemetry = Px4CommandAckTelemetry(
            timestamp_us=int(getattr(ack_message, "timestamp")),
            command=int(getattr(ack_message, "command")),
            result=int(getattr(ack_message, "result")),
            result_param1=int(getattr(ack_message, "result_param1", 0)),
            result_param2=int(getattr(ack_message, "result_param2", 0)),
        )
        self._latest_command_acks[telemetry.command] = telemetry
        return telemetry

    def validate_px4_control_state(self) -> None:
        """Raises on critical PX4 state errors once validation is enabled."""

        if not self.config.validate_px4_state:
            return

        state = self.control_state()
        if state.failsafe_active:
            raise RuntimeError("PX4 reports failsafe active; refusing further control commands.")

        if state.last_command_error is not None:
            raise RuntimeError(state.last_command_error)

        if (
            self.config.require_preflight_checks
            and self._offboard_setpoint_counter >= self.config.offboard_warmup_cycles
            and state.preflight_checks_known
            and not state.preflight_checks_pass
        ):
            raise RuntimeError("PX4 pre-flight checks are failing; refusing to arm into offboard mode.")

        if self._transition_started_us is None:
            return

        transition_deadline_us = self._transition_started_us + int(self.config.transition_timeout_s * 1_000_000)
        if self.now_us() < transition_deadline_us:
            return

        if not state.offboard_enabled:
            raise RuntimeError("PX4 did not confirm OFFBOARD mode before the transition timeout.")
        if not state.armed:
            raise RuntimeError("PX4 did not confirm arming before the transition timeout.")

    def _latest_command_error(self) -> str | None:
        """Returns a human-readable error for the latest arm/offboard acks."""

        watched_commands = (
            DictionaryMessageFactory.VEHICLE_CMD_DO_SET_MODE,
            DictionaryMessageFactory.VEHICLE_CMD_COMPONENT_ARM_DISARM,
        )
        rejected_results = {
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_DENIED,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_UNSUPPORTED,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_FAILED,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_CANCELLED,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_COMMAND_LONG_ONLY,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_COMMAND_INT_ONLY,
            DictionaryMessageFactory.VEHICLE_CMD_RESULT_UNSUPPORTED_MAV_FRAME,
        }
        for command in watched_commands:
            ack = self._latest_command_acks.get(command)
            if ack is not None and ack.result in rejected_results:
                return f"PX4 rejected command {ack.command} with result code {ack.result}."
        return None

    def now_us(self) -> int:
        """Returns the current time in microseconds."""

        return int(self._clock_ns() // 1_000)

    @staticmethod
    def ros_enu_to_px4_ned(vector_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        """Converts a ROS ENU world-frame vector to PX4 NED.

        PX4 expects world-frame fields in `TrajectorySetpoint` to use NED.
        Under the standard ENU->NED mapping:

        - North = ENU Y
        - East = ENU X
        - Down = -ENU Z
        """

        x_east, y_north, z_up = vector_xyz
        return (float(y_north), float(x_east), float(-z_up))

    @staticmethod
    def extract_timestamp_ns(image_message: Any) -> int:
        """Extracts a nanosecond timestamp from ROS-like message headers."""

        header = getattr(image_message, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is not None and hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        timestamp = getattr(image_message, "timestamp", None)
        if timestamp is not None:
            return int(timestamp)
        return 0

    @staticmethod
    def decode_image(image_message: Any) -> np.ndarray:
        """Decodes a ROS-style image message into an RGB `uint8` array."""

        height = int(getattr(image_message, "height"))
        width = int(getattr(image_message, "width"))
        encoding = str(getattr(image_message, "encoding", "rgb8")).lower()
        step = int(getattr(image_message, "step", width * 3))
        raw = np.frombuffer(getattr(image_message, "data"), dtype=np.uint8)

        if encoding in {"rgb8", "bgr8"}:
            channel_count = 3
            row_width = width * channel_count
            image = raw.reshape(height, step)[:, :row_width].reshape(height, width, channel_count)
            if encoding == "bgr8":
                image = image[:, :, ::-1]
            return np.ascontiguousarray(image)

        if encoding == "mono8":
            row_width = width
            mono = raw.reshape(height, step)[:, :row_width].reshape(height, width, 1)
            return np.repeat(mono, 3, axis=2)

        raise ValueError(f"Unsupported image encoding: {encoding}")

    @staticmethod
    def _publish(publisher: PublisherProtocol | None, message: Any) -> None:
        if publisher is not None:
            publisher.publish(message)
