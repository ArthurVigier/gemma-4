"""Benchmark export built from ARC metrics and PX4 supervision traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .arc_types import BenchmarkMetrics, BenchmarkTask
from .supervision import ControlEvent, ControlStateSnapshot


@dataclass(slots=True)
class SupervisionBenchmarkMetrics:
    """Aggregated supervision-side metrics for one benchmark episode."""

    snapshot_count: int
    telemetry_available_count: int
    ready_for_control_count: int
    armed_count: int
    offboard_count: int
    failsafe_count: int
    command_error_count: int
    event_count: int
    warning_event_count: int
    error_event_count: int
    offboard_transition_count: int
    arming_transition_count: int
    time_to_ready_ms: float | None
    mean_halt_probability: float | None


@dataclass(slots=True)
class BenchmarkEpisodeExport:
    """Export row that combines ARC task metrics and supervision metrics."""

    task_id: str
    task_family: str
    episode_steps: int
    success: bool
    symbolic_success: bool
    waypoint_success: bool
    termination_reason: str
    grid_accuracy: float
    action_accuracy: float
    latency_ms: float
    energy_joules: float
    within_budget: bool
    supervision: SupervisionBenchmarkMetrics


@dataclass(slots=True)
class BenchmarkSupervisionExporter:
    """Collects supervision snapshots/events and exports episode metrics."""

    snapshots: list[ControlStateSnapshot] = field(default_factory=list)
    events: list[ControlEvent] = field(default_factory=list)

    def record_snapshot(self, snapshot: ControlStateSnapshot) -> None:
        """Adds one supervision snapshot."""

        self.snapshots.append(snapshot)

    def record_event(self, event: ControlEvent) -> None:
        """Adds one structured control event."""

        self.events.append(event)

    def reset(self) -> None:
        """Clears the current episode trace."""

        self.snapshots.clear()
        self.events.clear()

    def supervision_metrics(self) -> SupervisionBenchmarkMetrics:
        """Computes supervision-side metrics from recorded traces."""

        telemetry_available_count = sum(1 for snapshot in self.snapshots if snapshot.telemetry_available)
        ready_for_control_count = sum(1 for snapshot in self.snapshots if snapshot.ready_for_control)
        armed_count = sum(1 for snapshot in self.snapshots if snapshot.armed)
        offboard_count = sum(1 for snapshot in self.snapshots if snapshot.offboard_enabled)
        failsafe_count = sum(1 for snapshot in self.snapshots if snapshot.failsafe_active)
        command_error_count = sum(1 for snapshot in self.snapshots if snapshot.last_command_error is not None)
        warning_event_count = sum(1 for event in self.events if event.severity == "warning")
        error_event_count = sum(1 for event in self.events if event.severity == "error")
        offboard_transition_count = sum(1 for event in self.events if event.event_type == "offboard_state_changed")
        arming_transition_count = sum(1 for event in self.events if event.event_type == "arming_state_changed")

        first_status_timestamp = next(
            (snapshot.status_timestamp_us for snapshot in self.snapshots if snapshot.status_timestamp_us is not None),
            None,
        )
        first_ready_timestamp = next(
            (
                snapshot.status_timestamp_us
                for snapshot in self.snapshots
                if snapshot.ready_for_control and snapshot.status_timestamp_us is not None
            ),
            None,
        )
        time_to_ready_ms = None
        if first_status_timestamp is not None and first_ready_timestamp is not None:
            time_to_ready_ms = (first_ready_timestamp - first_status_timestamp) / 1_000.0

        halt_probabilities = [
            snapshot.last_halt_probability
            for snapshot in self.snapshots
            if snapshot.last_halt_probability is not None
        ]
        mean_halt_probability = None
        if halt_probabilities:
            mean_halt_probability = float(sum(halt_probabilities) / len(halt_probabilities))

        return SupervisionBenchmarkMetrics(
            snapshot_count=len(self.snapshots),
            telemetry_available_count=telemetry_available_count,
            ready_for_control_count=ready_for_control_count,
            armed_count=armed_count,
            offboard_count=offboard_count,
            failsafe_count=failsafe_count,
            command_error_count=command_error_count,
            event_count=len(self.events),
            warning_event_count=warning_event_count,
            error_event_count=error_event_count,
            offboard_transition_count=offboard_transition_count,
            arming_transition_count=arming_transition_count,
            time_to_ready_ms=time_to_ready_ms,
            mean_halt_probability=mean_halt_probability,
        )

    def export_episode(
        self,
        *,
        task: BenchmarkTask,
        benchmark_metrics: BenchmarkMetrics,
        episode_steps: int,
        success: bool,
        symbolic_success: bool,
        waypoint_success: bool,
        termination_reason: str,
    ) -> BenchmarkEpisodeExport:
        """Builds the export row for one benchmark episode."""

        return BenchmarkEpisodeExport(
            task_id=task.task_id,
            task_family=task.family,
            episode_steps=episode_steps,
            success=success,
            symbolic_success=symbolic_success,
            waypoint_success=waypoint_success,
            termination_reason=termination_reason,
            grid_accuracy=benchmark_metrics.grid_accuracy,
            action_accuracy=benchmark_metrics.action_accuracy,
            latency_ms=benchmark_metrics.latency_ms,
            energy_joules=benchmark_metrics.energy_joules,
            within_budget=benchmark_metrics.within_budget,
            supervision=self.supervision_metrics(),
        )

    @staticmethod
    def export_to_jsonl(path: str | Path, rows: list[BenchmarkEpisodeExport]) -> Path:
        """Writes benchmark exports to a JSONL file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), sort_keys=True))
                handle.write("\n")
        return output_path

    @staticmethod
    def append_to_jsonl(path: str | Path, row: BenchmarkEpisodeExport) -> Path:
        """Appends one benchmark export row to a JSONL file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(row), sort_keys=True))
            handle.write("\n")
        return output_path


def benchmark_episode_to_json(row: BenchmarkEpisodeExport) -> str:
    """Serializes one benchmark episode export row."""

    return json.dumps(asdict(row), sort_keys=True)
