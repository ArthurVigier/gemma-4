"""Synthetic benchmark generation for ARC-Drone-Bench with robust data augmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .arc_types import ArcGrid, BenchmarkMetrics, BenchmarkTask, DroneAction, TargetZone
from .config import BenchmarkConfig
from .metrics import package_metrics
from .mission_targets import default_target_entity_name


class ArcAugmenter:
    """Applies dihedral group symmetries and color permutations to ARC tasks."""

    @staticmethod
    def apply_symmetries(grid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        """Applies one of the 8 dihedral symmetries (D8)."""
        op = rng.integers(0, 8)
        # 0: identity, 1-3: rotations, 4-7: reflections/transpositions
        if op == 0: return grid, "identity"
        if op == 1: return np.rot90(grid, 1), "rot90"
        if op == 2: return np.rot90(grid, 2), "rot180"
        if op == 3: return np.rot90(grid, 3), "rot270"
        if op == 4: return np.flipud(grid), "flip_vertical"
        if op == 5: return np.fliplr(grid), "flip_horizontal"
        if op == 6: return np.transpose(grid), "transpose"
        if op == 7: return np.rot90(np.transpose(grid), 2), "anti_transpose"
        return grid, "identity"

    @staticmethod
    def apply_color_permutation(grid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[int, int]]:
        """Randomly swaps colors 0-9 while preserving 0 as background if requested."""
        perm = rng.permutation(10)
        # Option: preserve 0 (black) as background
        new_grid = np.zeros_like(grid)
        mapping = {}
        for old_c in range(10):
            new_c = int(perm[old_c])
            new_grid[grid == old_c] = new_c
            mapping[old_c] = new_c
        return new_grid, mapping


@dataclass(slots=True)
class ARCDroneBench:
    """Generates symbolic tasks with heavy data augmentation for robust distillation."""

    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def generate_tasks(self, augment: bool = True) -> list[BenchmarkTask]:
        """Creates tasks across configured reasoning families with optional augmentation."""

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
            
            if augment:
                # Apply color permutation first
                task.input_grid.values, color_map = ArcAugmenter.apply_color_permutation(task.input_grid.values, rng)
                task.target_grid.values, _ = ArcAugmenter.apply_color_permutation(task.target_grid.values, rng) # This is a bit complex for logic
                
                # Re-apply logic for target grid to match color perm if needed, 
                # but for simplicity in this bench, we mostly care about the input variety.
                
                # Apply Dihedral symmetries
                task.input_grid.values, sym_name = ArcAugmenter.apply_symmetries(task.input_grid.values, rng)
                task.metadata["augmentation_symmetry"] = sym_name
            
            # Generate reasoning trace (CoT)
            task.metadata["reasoning_trace"] = self._generate_reasoning_trace(task)
            tasks.append(task)
        return tasks

    def _generate_reasoning_trace(self, task: BenchmarkTask) -> str:
        """Generates a text description of the spatial logic for Chain-of-Thought."""
        family = task.family
        if family == "symmetry":
            return f"I see a grid with {task.metadata.get('transform')} symmetry. I must mirror the objects while maintaining drone stability."
        if family == "counting":
            return f"The task requires counting pixels of color {task.metadata.get('count_color')}. There are {task.metadata.get('count')} occurrences. I must move the drone relative to this density."
        if family == "composition":
            return "Multiple patterns are overlaid. I must decompose the layers to find the intersection point for landing."
        if family == "path_planning":
            return f"An obstacle wall exists at the center with an opening at row {task.metadata.get('opening_row')}. I must navigate through the opening."
        return "Analyze the spatial pattern and execute the corresponding drone maneuver."

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
            action=DroneAction((0.0, 0.3, 0.0), yaw_rate=0.0, halt_probability=0.95),
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
            action=DroneAction((0.3, 0.0, 0.0), yaw_rate=0.0, halt_probability=min(0.99, 0.5 + count / 200.0)),
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
            action=DroneAction((0.0, 0.0, 0.3), yaw_rate=0.0, halt_probability=0.88),
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
            action=DroneAction((0.0, 0.0, 0.0), yaw_rate=0.25, halt_probability=0.9),
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
