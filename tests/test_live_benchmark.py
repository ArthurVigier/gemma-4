import json

import numpy as np

from arc_drone.arc_types import ArcGrid, DroneAction
from arc_drone.live_benchmark import LiveBenchmarkConfig, LiveBenchmarkRunner
from arc_drone.supervision import ControlEvent, ControlStateSnapshot


class _InferenceRecord:
    def __init__(self) -> None:
        self.action = DroneAction((0.3, 0.0, 0.0), yaw_rate=0.2, halt_probability=0.9)
        self.halted_at_step = 4
        self.predicted_grid = ArcGrid(np.zeros((30, 30), dtype=np.int64))


def _snapshot(ready: bool, ts_us: int) -> ControlStateSnapshot:
    return ControlStateSnapshot(
        simulator_name="gazebo_harmonic_px4_sitl",
        telemetry_available=True,
        preflight_checks_pass=True,
        preflight_checks_known=True,
        armed=True,
        offboard_enabled=ready,
        failsafe_active=False,
        ready_for_control=ready,
        last_command_error=None,
        status_timestamp_us=ts_us,
        nav_state=14 if ready else 0,
        arming_state=2,
        odometry_timestamp_us=ts_us,
        position_ned=(0.0, 0.0, -1.0),
        velocity_ned=(0.1, 0.0, 0.0),
        angular_velocity_frd=(0.0, 0.0, 0.0),
        last_halted_at_step=4,
        last_halt_probability=0.9,
        last_action_velocity_enu=(0.3, 0.0, 0.0),
        last_action_yaw_rate=0.2,
    )


def _snapshot_at_position(ready: bool, ts_us: int, position_ned: tuple[float, float, float]) -> ControlStateSnapshot:
    snapshot = _snapshot(ready, ts_us)
    snapshot.position_ned = position_ned
    return snapshot


def test_live_benchmark_runner_writes_episode_jsonl(tmp_path) -> None:
    runner = LiveBenchmarkRunner(
        LiveBenchmarkConfig(
            output_path=(tmp_path / "live_metrics.jsonl").as_posix(),
            task_count=4,
            max_episode_steps=10,
            ready_timeout_steps=10,
            rotate_max_rows_per_file=10,
        )
    )
    record = _InferenceRecord()
    record.action = runner.current_task.target_action
    record.predicted_grid = runner.current_task.target_grid
    event = ControlEvent(
        event_type="offboard_state_changed",
        severity="info",
        message="offboard_enabled=True",
        simulator_name="gazebo_harmonic_px4_sitl",
        status_timestamp_us=2_000,
        ready_for_control=True,
        armed=True,
        offboard_enabled=True,
        failsafe_active=False,
        last_command_error=None,
    )

    row0 = runner.record_tick(inference_record=record, snapshot=_snapshot(False, 1_000), events=[], latency_ms=25.0)
    row1 = runner.record_tick(
        inference_record=record,
        snapshot=_snapshot_at_position(True, 2_000, runner.current_task.target_zone.center_ned),
        events=[event],
        latency_ms=35.0,
    )

    assert row0 is None
    assert row1 is not None
    assert row1.success is True
    assert row1.symbolic_success is True
    assert row1.waypoint_success is True
    assert row1.termination_reason == "success"
    assert row1.supervision.snapshot_count == 2
    assert row1.supervision.time_to_ready_ms == 1.0

    payload = runner.active_output_path.read_text(encoding="utf-8").strip()
    decoded = json.loads(payload)
    assert decoded["task_id"].startswith(("symmetry-", "counting-", "composition-", "path_planning-"))
    assert decoded["supervision"]["event_count"] == 1
    assert runner.rows_written == 1


def test_live_benchmark_runner_rotates_output_files(tmp_path) -> None:
    runner = LiveBenchmarkRunner(
        LiveBenchmarkConfig(
            output_path=(tmp_path / "live_metrics.jsonl").as_posix(),
            task_count=4,
            max_episode_steps=1,
            ready_timeout_steps=5,
            rotate_max_rows_per_file=1,
        )
    )
    record = _InferenceRecord()
    record.action = runner.current_task.target_action

    row1 = runner.record_tick(
        inference_record=record,
        snapshot=_snapshot_at_position(True, 1_000, (99.0, 99.0, -99.0)),
        events=[],
        latency_ms=20.0,
    )
    first_path = runner.active_output_path
    row2 = runner.record_tick(
        inference_record=record,
        snapshot=_snapshot_at_position(True, 2_000, (99.0, 99.0, -99.0)),
        events=[],
        latency_ms=20.0,
    )
    second_path = runner.active_output_path

    assert row1 is not None
    assert row2 is not None
    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert first_path.name.endswith("part0001.jsonl")
    assert second_path.name.endswith("part0002.jsonl")
    assert row1.termination_reason == "max_steps_guard"


def test_live_benchmark_runner_ends_on_failsafe(tmp_path) -> None:
    runner = LiveBenchmarkRunner(
        LiveBenchmarkConfig(
            output_path=(tmp_path / "live_metrics.jsonl").as_posix(),
            task_count=4,
            max_episode_steps=10,
            ready_timeout_steps=10,
            rotate_max_rows_per_file=10,
        )
    )
    record = _InferenceRecord()
    failing_snapshot = ControlStateSnapshot(
        simulator_name="gazebo_harmonic_px4_sitl",
        telemetry_available=True,
        preflight_checks_pass=True,
        preflight_checks_known=True,
        armed=True,
        offboard_enabled=True,
        failsafe_active=True,
        ready_for_control=False,
        last_command_error=None,
        status_timestamp_us=3_000,
        nav_state=14,
        arming_state=2,
        odometry_timestamp_us=3_000,
        position_ned=(0.0, 0.0, -1.0),
        velocity_ned=(0.1, 0.0, 0.0),
        angular_velocity_frd=(0.0, 0.0, 0.0),
        last_halted_at_step=4,
        last_halt_probability=0.9,
        last_action_velocity_enu=(0.3, 0.0, 0.0),
        last_action_yaw_rate=0.2,
    )

    row = runner.record_tick(inference_record=record, snapshot=failing_snapshot, events=[], latency_ms=20.0)

    assert row is not None
    assert row.success is False
    assert row.termination_reason == "failsafe"


def test_live_benchmark_runner_requires_waypoint_hit_for_success(tmp_path) -> None:
    runner = LiveBenchmarkRunner(
        LiveBenchmarkConfig(
            output_path=(tmp_path / "live_metrics.jsonl").as_posix(),
            task_count=4,
            max_episode_steps=2,
            ready_timeout_steps=10,
            rotate_max_rows_per_file=10,
        )
    )
    record = _InferenceRecord()
    record.action = runner.current_task.target_action
    record.predicted_grid = runner.current_task.target_grid

    row0 = runner.record_tick(
        inference_record=record,
        snapshot=_snapshot_at_position(True, 1_000, (99.0, 99.0, -99.0)),
        events=[],
        latency_ms=20.0,
    )
    row1 = runner.record_tick(
        inference_record=record,
        snapshot=_snapshot_at_position(True, 2_000, (99.0, 99.0, -99.0)),
        events=[],
        latency_ms=20.0,
    )

    assert row0 is None
    assert row1 is not None
    assert row1.success is False
    assert row1.symbolic_success is True
    assert row1.waypoint_success is False
    assert row1.termination_reason == "max_steps_guard"
