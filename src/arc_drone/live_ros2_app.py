"""ROS 2 entrypoint used by the Gazebo/PX4 bringup launch file."""

from __future__ import annotations

import argparse
from time import perf_counter

try:  # pragma: no cover - only exercised in a live ROS 2 runtime
    import rclpy
except ImportError:  # pragma: no cover
    rclpy = None

from .gazebo_px4_adapter import GazeboPx4AdapterConfig, GazeboPx4Topics
from .live_benchmark import LiveBenchmarkConfig, LiveBenchmarkRunner
from .mission_targets import MissionTargetSubscription, MissionTargetTracker, default_mission_target_subscriptions
from .ros_node import ARCDroneNode


def parse_args() -> argparse.Namespace:
    """Parses runtime arguments for the live ROS 2 app."""

    parser = argparse.ArgumentParser(description="Run the ARC-drone ROS 2 control loop.")
    parser.add_argument("--node-name", default="arc_drone_node")
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--offboard-rate-hz", type=float, default=20.0)
    parser.add_argument("--benchmark-output-path", default="artifacts/benchmark/live_benchmark_metrics.jsonl")
    parser.add_argument("--benchmark-task-count", type=int, default=200)
    parser.add_argument("--benchmark-max-episode-steps", type=int, default=100)
    parser.add_argument("--benchmark-ready-timeout-steps", type=int, default=40)
    parser.add_argument("--benchmark-rotate-max-rows", type=int, default=1000)
    parser.add_argument(
        "--mission-marker-topic",
        action="append",
        default=[],
        help="Mission marker binding in the form entity_name=/ros/topic/odometry. Repeat as needed.",
    )
    return parser.parse_args()


def parse_mission_marker_topics(values: list[str]) -> list[MissionTargetSubscription]:
    """Parses CLI mission marker bindings."""

    if not values:
        return default_mission_target_subscriptions()

    subscriptions: list[MissionTargetSubscription] = []
    for value in values:
        entity_name, separator, topic_name = value.partition("=")
        if not separator or not entity_name or not topic_name:
            raise ValueError(
                f"Invalid mission marker mapping {value!r}. Expected entity_name=/ros/topic/odometry."
            )
        subscriptions.append(MissionTargetSubscription(entity_name=entity_name, topic_name=topic_name))
    return subscriptions


def main() -> None:
    """Runs the ROS 2 app and periodically steps the ARC-drone node."""

    if rclpy is None:
        raise RuntimeError("ROS 2 is not installed in this environment.")

    args = parse_args()
    rclpy.init()

    adapter_config = GazeboPx4AdapterConfig(
        topics=GazeboPx4Topics(image_topic=args.image_topic),
    )
    node = ARCDroneNode(name=args.node_name, adapter_config=adapter_config)
    mission_target_tracker = MissionTargetTracker(subscriptions=parse_mission_marker_topics(args.mission_marker_topic))
    mission_target_tracker.bind_ros_interfaces(node)
    benchmark_runner = LiveBenchmarkRunner(
        LiveBenchmarkConfig(
            output_path=args.benchmark_output_path,
            task_count=args.benchmark_task_count,
            max_episode_steps=args.benchmark_max_episode_steps,
            ready_timeout_steps=args.benchmark_ready_timeout_steps,
            rotate_max_rows_per_file=args.benchmark_rotate_max_rows,
        ),
        mission_target_tracker=mission_target_tracker,
    )
    period_s = 1.0 / args.offboard_rate_hz

    def _timer_callback() -> None:
        tick_started = perf_counter()
        try:
            record = node.step()
            snapshot = node.publish_control_state()
            events = node.last_control_events()
            node.log_px4_telemetry()
            row = benchmark_runner.record_tick(
                inference_record=record,
                snapshot=snapshot,
                events=events,
                latency_ms=(perf_counter() - tick_started) * 1_000.0,
            )
            if row is not None:
                node.publish_benchmark_metrics(row)
                node.get_logger().info(
                    f"Benchmark episode exported for task {row.task_id} to {benchmark_runner.active_output_path}"
                )
            node.get_logger().debug(
                f"Published action with halt probability {record.action.halt_probability:.3f}"
            )
        except RuntimeError as exc:
            # The simulator may still be starting up or waiting for the first image.
            node.publish_control_state()
            node.log_px4_telemetry()
            node.get_logger().debug(str(exc))

    node.create_timer(period_s, _timer_callback)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
