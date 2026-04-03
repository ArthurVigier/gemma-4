import json

import numpy as np

from arc_drone.arc_types import ArcGrid, BenchmarkMetrics, BenchmarkTask, DroneAction
from arc_drone.benchmark_export import BenchmarkSupervisionExporter, benchmark_episode_to_json
from arc_drone.supervision import ControlStateSnapshot


def test_benchmark_episode_to_json_serializes_episode_payload() -> None:
    exporter = BenchmarkSupervisionExporter()
    exporter.record_snapshot(
        ControlStateSnapshot(
            simulator_name="gazebo_harmonic_px4_sitl",
            telemetry_available=True,
            preflight_checks_pass=True,
            preflight_checks_known=True,
            armed=True,
            offboard_enabled=True,
            failsafe_active=False,
            ready_for_control=True,
            last_command_error=None,
            status_timestamp_us=1000,
            nav_state=14,
            arming_state=2,
            odometry_timestamp_us=1000,
            position_ned=(0.0, 0.0, -1.0),
            velocity_ned=(0.1, 0.0, 0.0),
            angular_velocity_frd=(0.0, 0.0, 0.0),
            last_halted_at_step=3,
            last_halt_probability=0.9,
            last_action_velocity_enu=(0.3, 0.0, 0.0),
            last_action_yaw_rate=0.2,
        )
    )
    task = BenchmarkTask(
        task_id="path_planning-0000",
        family="path_planning",
        input_grid=ArcGrid(np.zeros((2, 2), dtype=np.int64)),
        target_grid=ArcGrid(np.zeros((2, 2), dtype=np.int64)),
        target_action=DroneAction((0.3, 0.0, 0.0), yaw_rate=0.0, halt_probability=0.9),
    )
    row = exporter.export_episode(
        task=task,
        benchmark_metrics=BenchmarkMetrics(
            grid_accuracy=0.8,
            action_accuracy=1.0,
            latency_ms=35.0,
            energy_joules=1.2,
            within_budget=True,
        ),
        episode_steps=6,
        success=False,
        symbolic_success=False,
        waypoint_success=True,
        termination_reason="max_steps_guard",
    )

    payload = benchmark_episode_to_json(row)
    decoded = json.loads(payload)

    assert decoded["task_id"] == "path_planning-0000"
    assert decoded["latency_ms"] == 35.0
    assert decoded["waypoint_success"] is True
    assert decoded["termination_reason"] == "max_steps_guard"
    assert decoded["supervision"]["snapshot_count"] == 1
