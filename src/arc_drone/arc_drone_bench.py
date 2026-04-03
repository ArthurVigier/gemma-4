"""Synthetic benchmark generation for ARC-Drone-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arc_types import ArcGrid, BenchmarkMetrics, BenchmarkTask, DroneAction, TargetZone
from .config import BenchmarkConfig
from .metrics import package_metrics
from .mission_targets import default_target_entity_name


@dataclass(slots=True)
class ARCDroneBench:
    """Generates symbolic tasks that mimic abstract perception and control demands."""

    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def generate_tasks(self) -> list[BenchmarkTask]:
        """Creates at least `task_count` tasks across configured reasoning families."""

        rng = np.random.default_rng(self.config.seed)
        tasks: list[BenchmarkTask] = []
        family_generators = {
            "symmetry": self._make_symmetry_task,
            "counting": self._make_counting_task,
            "composition": self._make_composition_task,
            "path_planning": self._make_path_planning_task,
        }

        for index in range(self.config.task_count):
            family = self.config.task_families[index % len(self.config.task_families)]
            task = family_generators[family](rng, index)
            tasks.append(task)
        return tasks

    def evaluate_task(
        self,
        prediction: ArcGrid,
        predicted_action: DroneAction,
        task: BenchmarkTask,
        latency_ms: float,
        energy_joules: float,
    ) -> BenchmarkMetrics:
        """Evaluates one task under the benchmark budgets."""

        return package_metrics(
            prediction=prediction,
            target_grid=task.target_grid,
            predicted_action=predicted_action,
            target_action=task.target_action,
            latency_ms=latency_ms,
            energy_joules=energy_joules,
            config=self.config,
        )

    def _make_symmetry_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = self._random_grid(rng)
        target = np.fliplr(grid)
        return self._task(
            index=index,
            family="symmetry",
            input_grid=grid,
            target_grid=target,
            action=DroneAction((0.0, 0.4, 0.0), yaw_rate=0.2, halt_probability=0.95),
            target_zone=TargetZone(center_ned=(0.0, 1.2, -1.5), radius_m=0.45),
            target_entity_name=default_target_entity_name("symmetry"),
            metadata={"transform": "mirror_x"},
        )

    def _make_counting_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = self._random_grid(rng)
        highlighted_color = int(rng.integers(1, 9))
        count = int(np.sum(grid == highlighted_color))
        target = np.full_like(grid, fill_value=count % 10)
        return self._task(
            index=index,
            family="counting",
            input_grid=grid,
            target_grid=target,
            action=DroneAction((0.2, 0.0, 0.0), yaw_rate=0.0, halt_probability=min(0.99, 0.5 + count / 200.0)),
            target_zone=TargetZone(center_ned=(1.0, 0.0, -1.2), radius_m=0.4),
            target_entity_name=default_target_entity_name("counting"),
            metadata={"count_color": highlighted_color, "count": count},
        )

    def _make_composition_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = self._random_grid(rng)
        rotated = np.rot90(grid)
        target = np.where(rotated > 0, rotated, grid)
        return self._task(
            index=index,
            family="composition",
            input_grid=grid,
            target_grid=target,
            action=DroneAction((0.0, 0.0, 0.3), yaw_rate=-0.1, halt_probability=0.88),
            target_zone=TargetZone(center_ned=(0.0, 0.0, -2.0), radius_m=0.35),
            target_entity_name=default_target_entity_name("composition"),
            metadata={"transform": "rotate_plus_merge"},
        )

    def _make_path_planning_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.int64)
        grid[:, self.config.grid_width // 2] = 2
        opening = int(rng.integers(1, self.config.grid_height - 1))
        grid[opening - 1 : opening + 2, self.config.grid_width // 2] = 0
        grid[opening, 0] = 3
        target = grid.copy()
        target[opening, -1] = 4
        return self._task(
            index=index,
            family="path_planning",
            input_grid=grid,
            target_grid=target,
            action=DroneAction((0.3, 0.0, 0.0), yaw_rate=0.0, halt_probability=0.9),
            target_zone=TargetZone(center_ned=(1.8, 0.0, -1.5), radius_m=0.4),
            target_entity_name=default_target_entity_name("path_planning"),
            metadata={"opening_row": opening},
        )

    def _task(
        self,
        index: int,
        family: str,
        input_grid: np.ndarray,
        target_grid: np.ndarray,
        action: DroneAction,
        target_zone: TargetZone,
        target_entity_name: str | None,
        metadata: dict[str, str | int | float],
    ) -> BenchmarkTask:
        return BenchmarkTask(
            task_id=f"{family}-{index:04d}",
            family=family,
            input_grid=ArcGrid(input_grid.astype(np.int64)),
            target_grid=ArcGrid(target_grid.astype(np.int64)),
            target_action=action,
            target_zone=target_zone,
            target_entity_name=target_entity_name,
            metadata=metadata,
        )

    def _random_grid(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(
            low=0,
            high=10,
            size=(self.config.grid_height, self.config.grid_width),
            dtype=np.int64,
        )
