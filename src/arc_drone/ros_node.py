"""ROS2 integration point for the ARC-drone inference loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np
import torch

from .arc_types import ArcGrid, DroneAction
from .gazebo_px4_adapter import GazeboPx4Adapter, GazeboPx4AdapterConfig
from .benchmark_export import BenchmarkEpisodeExport, benchmark_episode_to_json
from .model import TRMReasoner
from .pipeline_vision import VisionGridConverter
from .simulators import DroneSimulator, SimObservation
from .supervision import (
    ControlEvent,
    ControlStateSnapshot,
    build_control_state_snapshot,
    control_event_to_json,
    control_state_events,
    control_state_transition_summary,
    snapshot_to_json,
    telemetry_log_line,
)

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - ROS2 is optional in local unit tests.
    rclpy = None
    Node = object
    String = None


@dataclass(slots=True)
class InferenceRecord:
    """Captures one full perception-reasoning-action pass."""

    action: DroneAction
    halted_at_step: int
    predicted_grid: ArcGrid


class ARCDroneNode(Node):
    """Reference ROS2 node for running the symbolic ARC-drone pipeline."""

    def __init__(
        self,
        name: str = "arc_drone_node",
        simulator: DroneSimulator | None = None,
        adapter_config: GazeboPx4AdapterConfig | None = None,
    ) -> None:
        if rclpy is None:
            raise RuntimeError("ROS2 is not installed in this environment.")
        super().__init__(name)
        self.converter = VisionGridConverter()
        self.reasoner = TRMReasoner().eval()
        self.simulator = simulator
        self.adapter_config = adapter_config or GazeboPx4AdapterConfig()
        self.last_inference_record: InferenceRecord | None = None
        self._last_control_state_snapshot: ControlStateSnapshot | None = None
        self._last_control_events: list[ControlEvent] = []
        self._last_telemetry_log_time_s = 0.0
        self.control_state_publisher = None
        self.control_events_publisher = None
        self.benchmark_metrics_publisher = None
        if String is not None:
            self.control_state_publisher = self.create_publisher(String, "arc_drone/control_state", 10)
            self.control_events_publisher = self.create_publisher(String, "arc_drone/control_events", 10)
            self.benchmark_metrics_publisher = self.create_publisher(String, "arc_drone/benchmark_metrics", 10)

        if self.simulator is None:
            try:
                gazebo_adapter = GazeboPx4Adapter(self.adapter_config)
                gazebo_adapter.bind_ros_interfaces(self)
                self.simulator = gazebo_adapter
            except RuntimeError:
                # The node can still be used in tests or custom deployments where
                # simulator interfaces are attached later.
                self.simulator = None

    def process_observation(self, observation: SimObservation) -> InferenceRecord:
        """Converts an RGB frame to an action through the TRM-like reasoner."""

        result = self.converter.convert_rgb(observation.rgb_image)
        grid = torch.from_numpy(result.grid.values).unsqueeze(0)
        with torch.no_grad():
            output = self.reasoner(grid)

        action_index = int(output.action_logits.argmax(dim=-1).item())
        action = self._decode_action(action_index, float(output.halt_probabilities[0, -1].item()))
        return InferenceRecord(
            action=action,
            halted_at_step=int(output.halted_at_step.item()),
            predicted_grid=result.grid,
        )

    def step(self) -> InferenceRecord:
        """Runs one full loop against the configured simulator adapter."""

        if self.simulator is None:
            raise RuntimeError("No simulator adapter is attached to this ROS2 node.")
        observation = self.simulator.get_observation()
        record = self.process_observation(observation)
        self.simulator.send_action(record.action)
        self.last_inference_record = record
        return record

    def build_control_state_snapshot(self) -> ControlStateSnapshot | None:
        """Builds the current supervision payload when using the PX4 adapter."""

        if not isinstance(self.simulator, GazeboPx4Adapter):
            return None
        return build_control_state_snapshot(
            simulator_name=self.adapter_config.simulator_name,
            control_state=self.simulator.control_state(),
            vehicle_status=self.simulator.latest_vehicle_status(),
            odometry=self.simulator.latest_odometry(),
            last_action=None if self.last_inference_record is None else self.last_inference_record.action,
            last_halted_at_step=None if self.last_inference_record is None else self.last_inference_record.halted_at_step,
        )

    def publish_control_state(self) -> ControlStateSnapshot | None:
        """Publishes the current control state on `arc_drone/control_state`."""

        snapshot = self.build_control_state_snapshot()
        if snapshot is None:
            return None

        if self.control_state_publisher is not None and String is not None:
            message = String()
            message.data = snapshot_to_json(snapshot)
            self.control_state_publisher.publish(message)

        events = control_state_events(self._last_control_state_snapshot, snapshot)
        self._last_control_events = events
        self.publish_control_events(events)
        transition_summary = control_state_transition_summary(self._last_control_state_snapshot, snapshot)
        if transition_summary is not None:
            self.get_logger().info(transition_summary)
        self._last_control_state_snapshot = snapshot
        return snapshot

    def last_control_events(self) -> list[ControlEvent]:
        """Returns the most recently published control events."""

        return list(self._last_control_events)

    def publish_control_events(self, events: list[ControlEvent]) -> None:
        """Publishes structured control events on `arc_drone/control_events`."""

        if self.control_events_publisher is None or String is None:
            return
        for event in events:
            message = String()
            message.data = control_event_to_json(event)
            self.control_events_publisher.publish(message)
            if event.severity == "error":
                self.get_logger().error(event.message)
            elif event.severity == "warning":
                self.get_logger().warning(event.message)
            else:
                self.get_logger().info(event.message)

    def publish_benchmark_metrics(self, row: BenchmarkEpisodeExport) -> None:
        """Publishes one benchmark episode export on `arc_drone/benchmark_metrics`."""

        if self.benchmark_metrics_publisher is None or String is None:
            return
        message = String()
        message.data = benchmark_episode_to_json(row)
        self.benchmark_metrics_publisher.publish(message)

    def log_px4_telemetry(self, min_period_s: float = 2.0) -> ControlStateSnapshot | None:
        """Logs a concise PX4 telemetry line at a bounded cadence."""

        snapshot = self.build_control_state_snapshot()
        if snapshot is None:
            return None

        now = monotonic()
        if now - self._last_telemetry_log_time_s >= min_period_s:
            self.get_logger().info(telemetry_log_line(snapshot))
            self._last_telemetry_log_time_s = now
        return snapshot

    @staticmethod
    def _decode_action(action_index: int, halt_probability: float) -> DroneAction:
        """Maps discrete action ids to a small fixed flight vocabulary."""

        vocabulary = [
            DroneAction((0.3, 0.0, 0.0), 0.0, halt_probability),
            DroneAction((-0.3, 0.0, 0.0), 0.0, halt_probability),
            DroneAction((0.0, 0.3, 0.0), 0.0, halt_probability),
            DroneAction((0.0, -0.3, 0.0), 0.0, halt_probability),
            DroneAction((0.0, 0.0, 0.3), 0.0, halt_probability),
            DroneAction((0.0, 0.0, -0.3), 0.0, halt_probability),
            DroneAction((0.0, 0.0, 0.0), 0.25, halt_probability),
            DroneAction((0.0, 0.0, 0.0), -0.25, halt_probability),
        ]
        return vocabulary[action_index]


def demo_observation(height: int = 120, width: int = 160) -> SimObservation:
    """Creates a deterministic fake observation for dry runs."""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = np.array([255, 65, 54], dtype=np.uint8)
    image[:, width // 2 :] = np.array([46, 204, 64], dtype=np.uint8)
    return SimObservation(rgb_image=image, timestamp_ns=0, simulator_name="dry-run")
