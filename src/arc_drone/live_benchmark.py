"""Live benchmark integration for automatic episode exports."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from .arc_drone_bench import ARCDroneBench
from .benchmark_export import BenchmarkEpisodeExport, BenchmarkSupervisionExporter
from .config import BenchmarkConfig
from .mission_targets import MissionTargetTracker
from .supervision import ControlEvent, ControlStateSnapshot


@dataclass(slots=True)
class LiveBenchmarkConfig:
    """Runtime settings for automatic live benchmark export."""

    output_path: str = "artifacts/benchmark/live_benchmark_metrics.jsonl"
    task_count: int = 200
    max_episode_steps: int = 100
    ready_timeout_steps: int = 40
    require_ready_for_success: bool = True
    end_on_success: bool = True
    end_on_failsafe: bool = True
    end_on_offboard_loss: bool = True
    end_on_command_error: bool = True
    flush_every_episode: bool = True
    rotate_max_rows_per_file: int = 1000
    timestamped_run_files: bool = True


@dataclass(slots=True)
class _LiveEpisodeState:
    """Mutable state for one live benchmark episode."""

    task_index: int
    steps: int = 0
    latency_ms_total: float = 0.0
    energy_joules_total: float = 0.0
    ready_seen: bool = False
    offboard_seen: bool = False
    waypoint_seen: bool = False
    closest_waypoint_distance_m: float | None = None


@dataclass(slots=True)
class LiveBenchmarkRunner:
    """Runs ARC-Drone-Bench episodes directly from live supervision/control ticks."""

    config: LiveBenchmarkConfig = field(default_factory=LiveBenchmarkConfig)
    benchmark_config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    mission_target_tracker: MissionTargetTracker = field(default_factory=MissionTargetTracker)
    bench: ARCDroneBench = field(init=False)
    tasks: list = field(init=False)
    exporter: BenchmarkSupervisionExporter = field(init=False)
    rows_written: int = field(init=False, default=0)
    file_rows_written: int = field(init=False, default=0)
    active_output_path: Path = field(init=False)
    run_id: str = field(init=False)
    _episode: _LiveEpisodeState = field(init=False)

    def __post_init__(self) -> None:
        bench_config = BenchmarkConfig(task_count=self.config.task_count)
        self.bench = ARCDroneBench(bench_config)
        self.tasks = self.bench.generate_tasks()
        self.exporter = BenchmarkSupervisionExporter()
        self.rows_written = 0
        self.file_rows_written = 0
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.active_output_path = self._resolve_output_path()
        self._episode = _LiveEpisodeState(task_index=0)

    @property
    def current_task(self):
        """Returns the task assigned to the current live episode."""

        return self.tasks[self._episode.task_index % len(self.tasks)]

    def record_tick(
        self,
        *,
        inference_record,
        snapshot: ControlStateSnapshot | None,
        events: list[ControlEvent],
        latency_ms: float,
    ) -> BenchmarkEpisodeExport | None:
        """Adds one live control tick and finalizes the episode when a terminal condition is met."""

        if snapshot is not None:
            self.exporter.record_snapshot(snapshot)
        for event in events:
            self.exporter.record_event(event)

        self._episode.steps += 1
        self._episode.latency_ms_total += latency_ms
        self._episode.energy_joules_total += self._estimate_energy_joules(
            latency_ms=latency_ms,
            velocity_xyz=inference_record.action.velocity_xyz,
            yaw_rate=inference_record.action.yaw_rate,
            ready_for_control=False if snapshot is None else snapshot.ready_for_control,
        )
        if snapshot is not None:
            self._episode.ready_seen = self._episode.ready_seen or snapshot.ready_for_control
            self._episode.offboard_seen = self._episode.offboard_seen or snapshot.offboard_enabled
            waypoint_distance_m = self._waypoint_distance_m(
                task=self.current_task,
                snapshot=snapshot,
            )
            if waypoint_distance_m is not None:
                if self._episode.closest_waypoint_distance_m is None:
                    self._episode.closest_waypoint_distance_m = waypoint_distance_m
                else:
                    self._episode.closest_waypoint_distance_m = min(
                        self._episode.closest_waypoint_distance_m,
                        waypoint_distance_m,
                    )
            self._episode.waypoint_seen = self._episode.waypoint_seen or self._waypoint_success(
                task=self.current_task,
                snapshot=snapshot,
                waypoint_distance_m=waypoint_distance_m,
            )

        task = self.current_task
        benchmark_metrics = self.bench.evaluate_task(
            prediction=inference_record.predicted_grid,
            predicted_action=inference_record.action,
            task=task,
            latency_ms=self._episode.latency_ms_total / self._episode.steps,
            energy_joules=self._episode.energy_joules_total,
        )
        symbolic_success = benchmark_metrics.grid_accuracy == 1.0 and benchmark_metrics.action_accuracy == 1.0
        waypoint_success = self._episode.waypoint_seen
        termination_reason = self._termination_reason(
            snapshot=snapshot,
            benchmark_metrics=benchmark_metrics,
            symbolic_success=symbolic_success,
            waypoint_success=waypoint_success,
        )
        if termination_reason is None:
            return None

        success = termination_reason == "success"
        row = self.exporter.export_episode(
            task=task,
            benchmark_metrics=benchmark_metrics,
            episode_steps=self._episode.steps,
            success=success,
            symbolic_success=symbolic_success,
            waypoint_success=waypoint_success,
            waypoint_distance_m=self._episode.closest_waypoint_distance_m,
            termination_reason=termination_reason,
        )
        if self.config.flush_every_episode:
            self.export_append(row)
        self.exporter.reset()
        self._episode = _LiveEpisodeState(task_index=self._episode.task_index + 1)
        return row

    def export_append(self, row: BenchmarkEpisodeExport) -> Path:
        """Appends one benchmark episode row to the configured JSONL output."""

        if self.file_rows_written >= self.config.rotate_max_rows_per_file:
            self.active_output_path = self._resolve_output_path()
            self.file_rows_written = 0

        output_path = self.active_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(self.row_to_json(row))
            handle.write("\n")
        self.rows_written += 1
        self.file_rows_written += 1
        return output_path

    @staticmethod
    def row_to_json(row: BenchmarkEpisodeExport) -> str:
        """Serializes one benchmark export row."""

        from dataclasses import asdict
        import json

        return json.dumps(asdict(row), sort_keys=True)

    @staticmethod
    def _estimate_energy_joules(
        *,
        latency_ms: float,
        velocity_xyz: tuple[float, float, float],
        yaw_rate: float,
        ready_for_control: bool,
    ) -> float:
        """Simple simulated energy proxy for one live step.

        The model stays intentionally lightweight and deterministic:
        base compute cost from latency plus a small actuation-dependent term.
        """

        velocity_norm = math.sqrt(sum(float(component) ** 2 for component in velocity_xyz))
        actuation_cost = 0.35 * velocity_norm + 0.1 * abs(float(yaw_rate))
        control_overhead = 0.15 if ready_for_control else 0.05
        compute_cost = 0.02 * (latency_ms / 10.0)
        return actuation_cost + control_overhead + compute_cost

    def _resolve_output_path(self) -> Path:
        """Resolves the active output path, applying run-based file rotation."""

        output_path = Path(self.config.output_path)
        if not self.config.timestamped_run_files:
            return output_path

        stem = output_path.stem
        suffix = output_path.suffix or ".jsonl"
        part_index = (self.rows_written // self.config.rotate_max_rows_per_file) + 1
        filename = f"{stem}_{self.run_id}_part{part_index:04d}{suffix}"
        return output_path.with_name(filename)

    def _termination_reason(
        self,
        *,
        snapshot: ControlStateSnapshot | None,
        benchmark_metrics,
        symbolic_success: bool,
        waypoint_success: bool,
    ) -> str | None:
        """Returns the terminal reason for the current episode, if any."""

        ready = False if snapshot is None else snapshot.ready_for_control
        offboard = False if snapshot is None else snapshot.offboard_enabled
        failsafe = False if snapshot is None else snapshot.failsafe_active
        command_error = None if snapshot is None else snapshot.last_command_error

        success = symbolic_success and waypoint_success
        if self.config.require_ready_for_success:
            success = success and ready
        if self.config.end_on_success and success:
            return "success"

        if self.config.end_on_failsafe and failsafe:
            return "failsafe"
        if self.config.end_on_command_error and command_error is not None:
            return "command_error"
        if self.config.end_on_offboard_loss and self._episode.offboard_seen and not offboard:
            return "offboard_lost"
        if not self._episode.ready_seen and self._episode.steps >= self.config.ready_timeout_steps:
            return "ready_timeout"
        if self._episode.steps >= self.config.max_episode_steps:
            return "max_steps_guard"
        return None

    def _waypoint_distance_m(
        self,
        *,
        task,
        snapshot: ControlStateSnapshot,
    ) -> float | None:
        """Returns the live distance from the drone to the tracked mission marker."""

        if task.target_entity_name is None or snapshot.position_ned is None:
            return None
        return self.mission_target_tracker.distance_to_target_ned(
            entity_name=task.target_entity_name,
            reference_position_ned=snapshot.position_ned,
        )

    @staticmethod
    def _waypoint_success(
        *,
        task,
        snapshot: ControlStateSnapshot,
        waypoint_distance_m: float | None,
    ) -> bool:
        """Returns whether the vehicle reached the live mission marker tolerance."""

        if task.target_zone is None or task.target_entity_name is None or snapshot.position_ned is None:
            return False
        if waypoint_distance_m is None:
            return False
        return waypoint_distance_m <= float(task.target_zone.radius_m)
