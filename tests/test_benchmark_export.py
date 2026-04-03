import json

import numpy as np

from arc_drone.arc_types import ArcGrid, BenchmarkMetrics, BenchmarkTask, DroneAction
from arc_drone.benchmark_export import BenchmarkSupervisionExporter
from arc_drone.supervision import ControlEvent, ControlStateSnapshot


def _snapshot(
    *,
    ready: bool,
    armed: bool,
    offboard: bool,
    status_timestamp_us: int,
    halt_probability: float | None = None,
    command_error: str | None = None,
) -> ControlStateSnapshot:
    return ControlStateSnapshot(
        simulator_name="gazebo_harmonic_px4_sitl",
        telemetry_available=True,
        preflight_checks_pass=True,
        preflight_checks_known=True,
        armed=armed,
        offboard_enabled=offboard,
        failsafe_active=False,
        ready_for_control=ready,
        last_command_error=command_error,
        status_timestamp_us=status_timestamp_us,
        nav_state=14 if offboard else 0,
        arming_state=2 if armed else 1,
        odometry_timestamp_us=status_timestamp_us,
        position_ned=(1.0, 2.0, -3.0),
        velocity_ned=(0.1, 0.2, -0.3),
        angular_velocity_frd=(0.01, 0.02, 0.03),
        last_halted_at_step=3,
        last_halt_probability=halt_probability,
        last_action_velocity_enu=(0.3, 0.0, 0.0),
        last_action_yaw_rate=0.25,
    )


def test_supervision_export_computes_time_to_ready_and_event_counts() -> None:
    exporter = BenchmarkSupervisionExporter()
    exporter.record_snapshot(_snapshot(ready=False, armed=False, offboard=False, status_timestamp_us=1_000_000))
    exporter.record_snapshot(_snapshot(ready=False, armed=True, offboard=False, status_timestamp_us=1_200_000))
    exporter.record_snapshot(_snapshot(ready=True, armed=True, offboard=True, status_timestamp_us=1_600_000, halt_probability=0.9))
    exporter.record_event(
        ControlEvent(
            event_type="arming_state_changed",
            severity="info",
            message="armed=True",
            simulator_name="gazebo_harmonic_px4_sitl",
            status_timestamp_us=1_200_000,
            ready_for_control=False,
            armed=True,
            offboard_enabled=False,
            failsafe_active=False,
            last_command_error=None,
        )
    )
    exporter.record_event(
        ControlEvent(
            event_type="offboard_state_changed",
            severity="info",
            message="offboard_enabled=True",
            simulator_name="gazebo_harmonic_px4_sitl",
            status_timestamp_us=1_600_000,
            ready_for_control=True,
            armed=True,
            offboard_enabled=True,
            failsafe_active=False,
            last_command_error=None,
        )
    )

    metrics = exporter.supervision_metrics()

    assert metrics.snapshot_count == 3
    assert metrics.ready_for_control_count == 1
    assert metrics.arming_transition_count == 1
    assert metrics.offboard_transition_count == 1
    assert metrics.time_to_ready_ms == 600.0
    assert metrics.mean_halt_probability == 0.9


def test_episode_export_and_jsonl_writer_roundtrip(tmp_path) -> None:
    exporter = BenchmarkSupervisionExporter()
    exporter.record_snapshot(_snapshot(ready=True, armed=True, offboard=True, status_timestamp_us=2_000_000, halt_probability=0.88))
    task = BenchmarkTask(
        task_id="symmetry-0001",
        family="symmetry",
        input_grid=ArcGrid(np.zeros((2, 2), dtype=np.int64)),
        target_grid=ArcGrid(np.zeros((2, 2), dtype=np.int64)),
        target_action=DroneAction((0.0, 0.4, 0.0), yaw_rate=0.2, halt_probability=0.95),
    )
    row = exporter.export_episode(
        task=task,
        benchmark_metrics=BenchmarkMetrics(
            grid_accuracy=1.0,
            action_accuracy=1.0,
            latency_ms=24.0,
            energy_joules=1.1,
            within_budget=True,
        ),
        episode_steps=5,
        success=True,
        symbolic_success=True,
        waypoint_success=True,
        waypoint_distance_m=0.12,
        termination_reason="success",
    )

    output_path = exporter.export_to_jsonl(tmp_path / "benchmark_metrics.jsonl", [row])
    payload = output_path.read_text(encoding="utf-8").strip()
    decoded = json.loads(payload)

    assert decoded["task_id"] == "symmetry-0001"
    assert decoded["success"] is True
    assert decoded["symbolic_success"] is True
    assert decoded["waypoint_success"] is True
    assert decoded["waypoint_distance_m"] == 0.12
    assert decoded["termination_reason"] == "success"
    assert decoded["supervision"]["snapshot_count"] == 1
    assert decoded["within_budget"] is True
