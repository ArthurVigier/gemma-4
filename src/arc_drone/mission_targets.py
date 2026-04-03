"""Mission target tracking backed by real Gazebo world entities or markers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - only exercised in a live ROS 2 runtime
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover
    Odometry = None


DEFAULT_FAMILY_TARGET_ENTITY_NAMES = {
    "symmetry": "arc_marker_symmetry",
    "counting": "arc_marker_counting",
    "composition": "arc_marker_composition",
    "path_planning": "arc_marker_path_planning",
}

DEFAULT_MISSION_TARGET_TOPICS = {
    "arc_marker_symmetry": "/arc_drone/mission_markers/arc_marker_symmetry/odometry",
    "arc_marker_counting": "/arc_drone/mission_markers/arc_marker_counting/odometry",
    "arc_marker_composition": "/arc_drone/mission_markers/arc_marker_composition/odometry",
    "arc_marker_path_planning": "/arc_drone/mission_markers/arc_marker_path_planning/odometry",
}


def default_target_entity_name(family: str) -> str | None:
    """Returns the default Gazebo mission target entity for one benchmark family."""

    return DEFAULT_FAMILY_TARGET_ENTITY_NAMES.get(family)


@dataclass(frozen=True, slots=True)
class MissionTargetSubscription:
    """ROS subscription metadata for one mission marker/entity."""

    entity_name: str
    topic_name: str


@dataclass(slots=True)
class MissionTargetState:
    """Latest tracked mission target pose from Gazebo."""

    entity_name: str
    source_topic: str
    timestamp_ns: int | None
    position_enu: tuple[float, float, float]
    position_ned: tuple[float, float, float]


def default_mission_target_subscriptions() -> list[MissionTargetSubscription]:
    """Returns the default mission marker subscriptions used by the live stack."""

    return [
        MissionTargetSubscription(entity_name=entity_name, topic_name=topic_name)
        for entity_name, topic_name in DEFAULT_MISSION_TARGET_TOPICS.items()
    ]


@dataclass(slots=True)
class MissionTargetTracker:
    """Tracks live Gazebo mission marker poses and computes success distances."""

    subscriptions: list[MissionTargetSubscription] = field(default_factory=default_mission_target_subscriptions)
    qos_depth: int = 10
    _latest_targets: dict[str, MissionTargetState] = field(init=False, default_factory=dict)
    _subscriptions: list[Any] = field(init=False, default_factory=list)

    def bind_ros_interfaces(self, ros_node: Any) -> None:
        """Creates ROS 2 odometry subscriptions for the configured mission targets."""

        if Odometry is None:
            raise RuntimeError("nav_msgs is not available in this environment.")

        self._subscriptions = [
            ros_node.create_subscription(
                Odometry,
                subscription.topic_name,
                lambda message, entity_name=subscription.entity_name, topic_name=subscription.topic_name: (
                    self.ingest_odometry_message(
                        entity_name=entity_name,
                        odometry_message=message,
                        source_topic=topic_name,
                    )
                ),
                self.qos_depth,
            )
            for subscription in self.subscriptions
        ]

    def update_target_position_ned(
        self,
        *,
        entity_name: str,
        position_ned: tuple[float, float, float],
        source_topic: str = "manual",
        timestamp_ns: int | None = None,
    ) -> MissionTargetState:
        """Updates one target pose directly in PX4 NED coordinates."""

        position_enu = self.ned_to_enu(position_ned)
        state = MissionTargetState(
            entity_name=entity_name,
            source_topic=source_topic,
            timestamp_ns=timestamp_ns,
            position_enu=position_enu,
            position_ned=tuple(float(component) for component in position_ned),
        )
        self._latest_targets[entity_name] = state
        return state

    def ingest_odometry_message(
        self,
        *,
        entity_name: str,
        odometry_message: Any,
        source_topic: str | None = None,
    ) -> MissionTargetState:
        """Ingests one ROS odometry message for a Gazebo mission marker/entity."""

        pose_with_covariance = getattr(odometry_message, "pose", None)
        pose = getattr(pose_with_covariance, "pose", pose_with_covariance)
        position = getattr(pose, "position")
        position_enu = (
            float(getattr(position, "x")),
            float(getattr(position, "y")),
            float(getattr(position, "z")),
        )
        header = getattr(odometry_message, "header", None)
        stamp = None if header is None else getattr(header, "stamp", None)
        timestamp_ns = self._timestamp_ns_from_stamp(stamp)
        state = MissionTargetState(
            entity_name=entity_name,
            source_topic=source_topic or "",
            timestamp_ns=timestamp_ns,
            position_enu=position_enu,
            position_ned=self.enu_to_ned(position_enu),
        )
        self._latest_targets[entity_name] = state
        return state

    def latest_target(self, entity_name: str) -> MissionTargetState | None:
        """Returns the latest tracked state for one mission target."""

        return self._latest_targets.get(entity_name)

    def target_position_ned(self, entity_name: str) -> tuple[float, float, float] | None:
        """Returns the latest target pose in PX4 NED coordinates."""

        state = self.latest_target(entity_name)
        return None if state is None else state.position_ned

    def distance_to_target_ned(
        self,
        *,
        entity_name: str,
        reference_position_ned: tuple[float, float, float],
    ) -> float | None:
        """Returns the distance from a PX4 NED position to a tracked target."""

        target_position_ned = self.target_position_ned(entity_name)
        if target_position_ned is None:
            return None

        dx = float(reference_position_ned[0]) - float(target_position_ned[0])
        dy = float(reference_position_ned[1]) - float(target_position_ned[1])
        dz = float(reference_position_ned[2]) - float(target_position_ned[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def is_target_reached(
        self,
        *,
        entity_name: str,
        reference_position_ned: tuple[float, float, float],
        tolerance_m: float,
    ) -> bool:
        """Returns whether the tracked mission marker is inside the success radius."""

        distance_m = self.distance_to_target_ned(
            entity_name=entity_name,
            reference_position_ned=reference_position_ned,
        )
        if distance_m is None:
            return False
        return distance_m <= float(tolerance_m)

    @staticmethod
    def enu_to_ned(position_enu: tuple[float, float, float]) -> tuple[float, float, float]:
        """Converts ROS/Gazebo ENU coordinates to PX4 NED coordinates."""

        east, north, up = (float(component) for component in position_enu)
        return (north, east, -up)

    @staticmethod
    def ned_to_enu(position_ned: tuple[float, float, float]) -> tuple[float, float, float]:
        """Converts PX4 NED coordinates back to ROS/Gazebo ENU coordinates."""

        north, east, down = (float(component) for component in position_ned)
        return (east, north, -down)

    @staticmethod
    def _timestamp_ns_from_stamp(stamp: Any) -> int | None:
        """Extracts a ROS2-style timestamp from a `builtin_interfaces/Time` object."""

        if stamp is None:
            return None
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        return int(sec) * 1_000_000_000 + int(nanosec)
