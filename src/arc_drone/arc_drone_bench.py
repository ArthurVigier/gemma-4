"""Synthetic benchmark generation for ARC-Drone-Bench with Isaac Sim Scene Descriptors."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .arc_types import ArcGrid, BenchmarkTask, DroneAction, TargetZone
from .config import BenchmarkConfig
from .mission_targets import default_target_entity_name


@dataclass(slots=True)
class IsaacEntity:
    """Represents a 3D entity in the Isaac Sim environment."""
    prim_type: str  # e.g., "cube", "sphere", "xform"
    pos_ned: tuple[float, float, float]
    scale: tuple[float, float, float]
    color_id: int
    material_category: str = "plastic"  # For Replicator domain randomization
    semantic_label: str = "arc_object"


class ARCDroneBench:
    """Generates tasks with both symbolic grids and Isaac Sim scene descriptors."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()
        self.repo_root = Path(__file__).resolve().parents[2]
        # Scale: how many meters is one grid cell?
        self.cell_size_m: float = 0.15

        # ARC Value to Isaac Material Mapping
        self.material_map = {
            0: "void",
            1: "metal",      # Blue
            2: "concrete",   # Red
            3: "plastic",    # Green
            4: "rubber",     # Yellow
            5: "wall_paint", # Gray
            6: "emissive",   # Magenta
            7: "fabric",     # Orange
            8: "glass",      # Azure
            9: "carbon",     # Maroon
        }

    def generate_tasks(self, augment: bool = True) -> list[BenchmarkTask]:
        """Creates tasks with rich 3D scene metadata for NVIDIA Replicator."""
        rng = np.random.default_rng(self.config.seed)
        synthetic_tasks: list[BenchmarkTask] = []
        family_generators = {
            "symmetry": self._make_symmetry_task,
            "counting": self._make_counting_task,
            "composition": self._make_composition_task,
            "path_planning": self._make_path_planning_task,
        }

        for index in range(self.config.task_count):
            family = self.config.task_families[index % len(self.config.task_families)]
            task = family_generators[family](rng, index)

            augmentations: list[str] = []
            if augment:
                augmentations = self._apply_synthetic_augmentation(task, rng)
            self._finalize_task_metadata(task, rng=rng, source_type="synthetic", augmentations=augmentations)
            synthetic_tasks.append(task)

        real_tasks = self._load_real_tasks(limit=self.config.task_count)
        if not real_tasks:
            return synthetic_tasks

        ratio = min(max(float(self.config.real_data_ratio), 0.0), 1.0)
        if ratio <= 0.0:
            return synthetic_tasks

        real_count = min(len(real_tasks), int(round(self.config.task_count * ratio)))
        if real_count <= 0:
            return synthetic_tasks

        synthetic_count = max(self.config.task_count - real_count, 0)
        mixed_tasks = synthetic_tasks[:synthetic_count] + real_tasks[:real_count]
        rng.shuffle(mixed_tasks)
        return mixed_tasks[:self.config.task_count]

    def _load_real_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        if self.config.real_dataset:
            return self._load_hugging_face_tasks(limit=limit)

        manifest_path = self._resolve_real_data_path()
        if manifest_path is None:
            return []

        records = self._read_manifest_records(manifest_path)
        rng = np.random.default_rng(self.config.seed + 10_000)
        rng.shuffle(records)
        tasks: list[BenchmarkTask] = []
        for index, record in enumerate(records):
            if limit is not None and index >= limit:
                break
            task = self._task_from_record(record, index=index, manifest_path=manifest_path)
            self._finalize_task_metadata(task, rng=rng, source_type="real", augmentations=[])
            tasks.append(task)
        return tasks

    def _load_hugging_face_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Loading a real dataset from the Hugging Face Hub requires the 'datasets' package."
            ) from exc

        dataset_id = self._resolve_real_dataset_id()
        split = self.config.real_dataset_split
        dataset = load_dataset(dataset_id, split=split, streaming=True)
        dataset = dataset.shuffle(buffer_size=10_000, seed=self.config.seed + 20_000)

        rows: list[dict[str, Any]] = []
        max_items = limit if limit is not None else self.config.task_count
        for index, row in enumerate(dataset):
            if index >= max_items:
                break
            rows.append(row)

        rng = np.random.default_rng(self.config.seed + 30_000)
        tasks: list[BenchmarkTask] = []
        for index, row in enumerate(rows):
            task = self._task_from_hf_row(
                row,
                index=index,
                dataset_id=dataset_id,
                dataset_name=self.config.real_dataset or dataset_id,
                rng=rng,
            )
            self._finalize_task_metadata(task, rng=rng, source_type="real_hf", augmentations=[])
            tasks.append(task)
        return tasks

    def _resolve_real_data_path(self) -> Path | None:
        if self.config.real_data_path:
            candidate = Path(self.config.real_data_path).expanduser()
            return candidate.resolve() if candidate.exists() else None

        if not self.config.auto_discover_real_data:
            return None

        env_candidates = [
            os.environ.get("ARC_DRONE_REAL_DATA_PATH"),
            os.environ.get("ARC_DRONE_REAL_MANIFEST"),
        ]
        path_candidates = [
            self.repo_root / "data" / "arc_drone_real_manifest.jsonl",
            self.repo_root / "data" / "arc_drone_real_manifest.json",
            self.repo_root / "artifacts" / "datasets" / "arc_drone_real_manifest.jsonl",
            self.repo_root / "artifacts" / "datasets" / "arc_drone_real_manifest.json",
            Path("/workspace/gemma-4/data/arc_drone_real_manifest.jsonl"),
            Path("/workspace/gemma-4/data/arc_drone_real_manifest.json"),
        ]

        for raw_candidate in env_candidates:
            if not raw_candidate:
                continue
            candidate = Path(raw_candidate).expanduser()
            if candidate.exists():
                return candidate.resolve()

        for candidate in path_candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _resolve_real_dataset_id(self) -> str:
        aliases = {
            "visdrone": "Voxel51/VisDrone2019-DET",
            "visdrone_det": "Voxel51/VisDrone2019-DET",
            "drone_detection": "pathikg/drone-detection-dataset",
            "pathikg_drone_detection": "pathikg/drone-detection-dataset",
        }
        assert self.config.real_dataset is not None
        return aliases.get(self.config.real_dataset, self.config.real_dataset)

    def _task_from_hf_row(
        self,
        row: dict[str, Any],
        *,
        index: int,
        dataset_id: str,
        dataset_name: str,
        rng: np.random.Generator,
    ) -> BenchmarkTask:
        pil_image = self._extract_row_image(row)
        input_grid = self._hf_row_to_grid(row, pil_image=pil_image)
        target_grid = input_grid.copy()
        target_action = self._hf_row_to_action(row, input_grid=input_grid)
        target_zone = self._hf_row_to_target_zone(input_grid)
        metadata: dict[str, Any] = {
            "pil_image": pil_image,
            "hf_dataset_id": dataset_id,
            "hf_dataset_name": dataset_name,
            "hf_split": self.config.real_dataset_split,
            "hf_row_keys": sorted(str(key) for key in row.keys()),
        }

        objects = row.get("objects")
        if isinstance(objects, dict):
            metadata["hf_object_count"] = len(objects.get("bbox", []))
            metadata["isaac_scene"] = self._scene_from_detection_row(
                grid=input_grid,
                objects=objects,
                width=row.get("width"),
                height=row.get("height"),
                task_id=f"{dataset_name}-{index:04d}",
                rng=rng,
            )
            metadata["reasoning_trace"] = self._reasoning_trace_from_detection_row(objects=objects, input_grid=input_grid)

        return BenchmarkTask(
            task_id=f"{dataset_name}-{index:04d}",
            family=f"real_{self._slugify(dataset_name)}",
            input_grid=ArcGrid(input_grid),
            target_grid=ArcGrid(target_grid),
            target_action=target_action,
            target_zone=target_zone,
            target_entity_name=None,
            metadata=metadata,
        )

    def _read_manifest_records(self, manifest_path: Path) -> list[dict[str, Any]]:
        suffix = manifest_path.suffix.lower()
        raw_text = manifest_path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            return [json.loads(line) for line in raw_text.splitlines() if line.strip()]

        payload = json.loads(raw_text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            records = payload.get("tasks")
            if isinstance(records, list):
                return records
        raise ValueError(f"Unsupported real-data manifest structure in {manifest_path}")

    def _task_from_record(self, record: dict[str, Any], *, index: int, manifest_path: Path) -> BenchmarkTask:
        family = str(record.get("family", "real_world"))
        input_grid = self._coerce_grid(record["input_grid"])
        target_grid = self._coerce_grid(record.get("target_grid", record["input_grid"]))
        action = self._coerce_action(record)
        target_zone = self._coerce_target_zone(record.get("target_zone"))
        metadata = dict(record.get("metadata", {}))

        image_path = record.get("image_path")
        if image_path:
            metadata["image_path"] = str((manifest_path.parent / str(image_path)).resolve())

        if "isaac_scene" in record:
            metadata["isaac_scene"] = record["isaac_scene"]
        if "reasoning_trace" in record:
            metadata["reasoning_trace"] = str(record["reasoning_trace"])
        metadata["real_manifest"] = str(manifest_path)

        return BenchmarkTask(
            task_id=str(record.get("task_id", f"real-{family}-{index:04d}")),
            family=family,
            input_grid=ArcGrid(input_grid),
            target_grid=ArcGrid(target_grid),
            target_action=action,
            target_zone=target_zone,
            target_entity_name=record.get("target_entity_name"),
            metadata=metadata,
        )

    def _coerce_grid(self, value: Any) -> np.ndarray:
        grid = np.asarray(value, dtype=np.int64)
        if grid.ndim != 2:
            raise ValueError("Real-data grid must be a 2D array.")
        if int(grid.min()) < 0 or int(grid.max()) > 9:
            raise ValueError("Real-data grid values must stay in the [0, 9] range.")
        return grid

    def _coerce_action(self, record: dict[str, Any]) -> DroneAction:
        action_payload = record.get("target_action")
        if isinstance(action_payload, dict):
            velocity = tuple(float(v) for v in action_payload["velocity_xyz"])
            yaw_rate = float(action_payload.get("yaw_rate", 0.0))
            halt_probability = float(action_payload["halt_probability"])
            return DroneAction(velocity, yaw_rate, halt_probability)

        action_index = int(record["action_index"])
        halt_step = int(record["halt_step"])
        actions = {
            0: DroneAction((0.3, 0.0, 0.0), 0.0, 0.90),
            1: DroneAction((-0.3, 0.0, 0.0), 0.0, 0.90),
            2: DroneAction((0.0, 0.3, 0.0), 0.0, 0.95),
            3: DroneAction((0.0, -0.3, 0.0), 0.0, 0.95),
            4: DroneAction((0.0, 0.0, 0.3), 0.0, 0.88),
            5: DroneAction((0.0, 0.0, -0.3), 0.0, 0.88),
            6: DroneAction((0.0, 0.0, 0.0), 0.25, 0.90),
            7: DroneAction((0.0, 0.0, 0.0), -0.25, 0.90),
        }
        action = actions.get(action_index)
        if action is None:
            raise ValueError(f"Unsupported action_index {action_index} in real-data manifest.")
        halt_probability = min(max(halt_step / 6.0, 0.0), 1.0)
        return DroneAction(action.velocity_xyz, action.yaw_rate, halt_probability)

    def _coerce_target_zone(self, payload: Any) -> TargetZone | None:
        if payload is None:
            return None
        center_ned = tuple(float(v) for v in payload["center_ned"])
        radius_m = float(payload["radius_m"])
        return TargetZone(center_ned, radius_m)

    def _extract_row_image(self, row: dict[str, Any]) -> Image.Image:
        image = row.get("image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if hasattr(image, "convert"):
            return image.convert("RGB")
        raise ValueError("Expected an image-like field in the real dataset row.")

    def _hf_row_to_grid(self, row: dict[str, Any], *, pil_image: Image.Image) -> np.ndarray:
        if isinstance(row.get("objects"), dict):
            return self._grid_from_detection_row(
                width=int(row.get("width", pil_image.width)),
                height=int(row.get("height", pil_image.height)),
                objects=row["objects"],
            )
        return self._grid_from_image(pil_image)

    def _grid_from_detection_row(self, *, width: int, height: int, objects: dict[str, Any]) -> np.ndarray:
        grid = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.int64)
        bboxes = list(objects.get("bbox", []))
        categories = list(objects.get("category", []))
        for idx, bbox in enumerate(bboxes):
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, box_w, box_h = [float(value) for value in bbox]
            category = int(categories[idx]) if idx < len(categories) else idx + 1
            color_value = (category % 9) + 1

            x0 = int(np.clip(np.floor((x / max(width, 1)) * self.config.grid_width), 0, self.config.grid_width - 1))
            y0 = int(np.clip(np.floor((y / max(height, 1)) * self.config.grid_height), 0, self.config.grid_height - 1))
            x1 = int(np.clip(np.ceil(((x + box_w) / max(width, 1)) * self.config.grid_width), x0 + 1, self.config.grid_width))
            y1 = int(np.clip(np.ceil(((y + box_h) / max(height, 1)) * self.config.grid_height), y0 + 1, self.config.grid_height))
            grid[y0:y1, x0:x1] = color_value
        return grid

    def _grid_from_image(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.config.grid_width, self.config.grid_height))
        arr = np.asarray(resized, dtype=np.float32)
        channel_bins = np.floor(arr / 85.3333333333).astype(np.int64)
        grid = (channel_bins[:, :, 0] + 3 * channel_bins[:, :, 1] + 5 * channel_bins[:, :, 2]) % 10
        return grid.astype(np.int64)

    def _hf_row_to_action(self, row: dict[str, Any], *, input_grid: np.ndarray) -> DroneAction:
        if isinstance(row.get("objects"), dict):
            return self._action_from_detection_row(
                width=float(row.get("width", 1)),
                height=float(row.get("height", 1)),
                objects=row["objects"],
            )
        return self._action_from_grid_centroid(input_grid)

    def _action_from_detection_row(self, *, width: float, height: float, objects: dict[str, Any]) -> DroneAction:
        bboxes = list(objects.get("bbox", []))
        if not bboxes:
            return DroneAction((0.0, 0.0, 0.0), 0.0, 0.98)

        centers_x = []
        centers_y = []
        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, box_w, box_h = [float(value) for value in bbox]
            centers_x.append(x + box_w / 2.0)
            centers_y.append(y + box_h / 2.0)
        if not centers_x:
            return DroneAction((0.0, 0.0, 0.0), 0.0, 0.98)

        mean_x = float(np.mean(centers_x))
        mean_y = float(np.mean(centers_y))
        x_bias = (mean_x / max(width, 1.0)) - 0.5
        y_bias = 0.5 - (mean_y / max(height, 1.0))
        halt_probability = min(0.99, 0.70 + 0.03 * len(centers_x))

        if abs(x_bias) >= abs(y_bias):
            return DroneAction((0.3 if x_bias >= 0 else -0.3, 0.0, 0.0), 0.0, halt_probability)
        return DroneAction((0.0, 0.3 if y_bias >= 0 else -0.3, 0.0), 0.0, halt_probability)

    def _action_from_grid_centroid(self, grid: np.ndarray) -> DroneAction:
        ys, xs = np.nonzero(grid > 0)
        if len(xs) == 0:
            return DroneAction((0.0, 0.0, 0.0), 0.0, 0.98)

        x_bias = (float(xs.mean()) / max(grid.shape[1] - 1, 1)) - 0.5
        y_bias = 0.5 - (float(ys.mean()) / max(grid.shape[0] - 1, 1))
        halt_probability = min(0.99, 0.72 + 0.001 * len(xs))

        if abs(x_bias) >= abs(y_bias):
            return DroneAction((0.3 if x_bias >= 0 else -0.3, 0.0, 0.0), 0.0, halt_probability)
        return DroneAction((0.0, 0.3 if y_bias >= 0 else -0.3, 0.0), 0.0, halt_probability)

    def _hf_row_to_target_zone(self, grid: np.ndarray) -> TargetZone:
        ys, xs = np.nonzero(grid > 0)
        if len(xs) == 0:
            return TargetZone((0.0, 0.0, -1.5), 0.45)
        mean_y = float(np.mean(ys))
        mean_x = float(np.mean(xs))
        north, east, _ = self._grid_to_ned(int(round(mean_y)), int(round(mean_x)), z=-1.5)
        return TargetZone((north, east, -1.5), 0.45)

    def _scene_from_detection_row(
        self,
        *,
        grid: np.ndarray,
        objects: dict[str, Any],
        width: Any,
        height: Any,
        task_id: str,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        entities: list[dict[str, Any]] = []
        bbox_list = list(objects.get("bbox", []))
        categories = list(objects.get("category", []))
        width_f = float(width or 1.0)
        height_f = float(height or 1.0)
        for idx, bbox in enumerate(bbox_list[:128]):
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, box_w, box_h = [float(value) for value in bbox]
            center_x = x + box_w / 2.0
            center_y = y + box_h / 2.0
            grid_x = int(np.clip(round((center_x / max(width_f, 1.0)) * (self.config.grid_width - 1)), 0, self.config.grid_width - 1))
            grid_y = int(np.clip(round((center_y / max(height_f, 1.0)) * (self.config.grid_height - 1)), 0, self.config.grid_height - 1))
            pos = self._grid_to_ned(grid_y, grid_x)
            category = int(categories[idx]) if idx < len(categories) else idx + 1
            entities.append(
                {
                    "prim_type": "cube",
                    "position": pos,
                    "scale": (
                        max(self.cell_size_m * 0.5, (box_h / max(height_f, 1.0)) * self.config.grid_height * self.cell_size_m),
                        max(self.cell_size_m * 0.5, (box_w / max(width_f, 1.0)) * self.config.grid_width * self.cell_size_m),
                        self.cell_size_m,
                    ),
                    "color_id": (category % 9) + 1,
                    "material": self.material_map.get((category % 9) + 1, "plastic"),
                    "semantics": {"class": "detected_object", "id": f"det_{idx}", "category": category},
                }
            )

        scene = self._build_scene_descriptor(
            BenchmarkTask(
                task_id=task_id,
                family="real_detection",
                input_grid=ArcGrid(grid.astype(np.int64)),
                target_grid=ArcGrid(grid.astype(np.int64)),
                target_action=DroneAction((0.0, 0.0, 0.0), 0.0, 0.9),
                metadata={},
            ),
            rng,
        )
        if entities:
            scene["entities"] = entities
        return scene

    def _reasoning_trace_from_detection_row(self, *, objects: dict[str, Any], input_grid: np.ndarray) -> str:
        categories = list(objects.get("category", []))
        unique_categories = len({int(category) for category in categories})
        object_count = len(list(objects.get("bbox", [])))
        occupied_cells = int(np.count_nonzero(input_grid))
        return (
            "Reasoning: Real drone scene. "
            f"I detect {object_count} objects across {unique_categories} categories, project them into the logical grid, "
            f"and use the occupied-cell structure ({occupied_cells} active cells) to choose the drone action."
        )

    def _slugify(self, value: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_")

    def _grid_to_ned(self, y: int, x: int, z: float = 0.0) -> tuple[float, float, float]:
        """Converts grid indices to Isaac Sim North-East-Down (NED) meters.
        Center of grid is (0,0) in NED.
        """
        center_y = self.config.grid_height / 2
        center_x = self.config.grid_width / 2
        north = (center_y - y) * self.cell_size_m
        east = (x - center_x) * self.cell_size_m
        return (north, east, z)

    def _finalize_task_metadata(
        self,
        task: BenchmarkTask,
        *,
        rng: np.random.Generator,
        source_type: str,
        augmentations: list[str],
    ) -> None:
        task.metadata["source_type"] = f"{source_type}_augmented" if augmentations else source_type
        task.metadata["dataset_version"] = self.config.dataset_version
        task.metadata["augmentations"] = list(augmentations)
        task.metadata.setdefault("isaac_scene", self._build_scene_descriptor(task, rng))
        task.metadata.setdefault("reasoning_trace", self._generate_reasoning_trace(task, rng))

    def _apply_synthetic_augmentation(self, task: BenchmarkTask, rng: np.random.Generator) -> list[str]:
        augmentations: list[str] = []

        if rng.random() < 0.85:
            color_mapping = self._sample_color_mapping(rng)
            self._apply_color_permutation(task, color_mapping)
            task.metadata["color_map"] = color_mapping
            augmentations.append("color_permutation")

        if rng.random() < 0.35:
            dy = int(rng.integers(-2, 3))
            dx = int(rng.integers(-2, 3))
            if dy != 0 or dx != 0:
                self._apply_translation(task, dy=dy, dx=dx)
                augmentations.append(f"translate({dy},{dx})")

        return augmentations

    def _sample_color_mapping(self, rng: np.random.Generator) -> dict[int, int]:
        colors = np.arange(1, 10, dtype=np.int64)
        shuffled = colors.copy()
        rng.shuffle(shuffled)
        mapping = {0: 0}
        mapping.update({int(old): int(new) for old, new in zip(colors, shuffled, strict=True)})
        return mapping

    def _apply_color_permutation(self, task: BenchmarkTask, mapping: dict[int, int]) -> None:
        task.input_grid.values = self._remap_grid_colors(task.input_grid.values, mapping)
        if task.family not in {"counting", "path_planning"}:
            task.target_grid.values = self._remap_grid_colors(task.target_grid.values, mapping)
        count_color = task.metadata.get("count_color")
        if isinstance(count_color, (int, np.integer)):
            task.metadata["count_color"] = mapping.get(int(count_color), int(count_color))

    def _remap_grid_colors(self, grid: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
        lut = np.arange(10, dtype=np.int64)
        for old_value, new_value in mapping.items():
            lut[int(old_value)] = int(new_value)
        return lut[grid]

    def _apply_translation(self, task: BenchmarkTask, *, dy: int, dx: int) -> None:
        task.input_grid.values = self._translate_grid(task.input_grid.values, dy=dy, dx=dx)
        task.target_grid.values = self._translate_grid(task.target_grid.values, dy=dy, dx=dx)
        if task.target_zone is not None:
            task.target_zone = self._shift_target_zone(task.target_zone, dy=dy, dx=dx)
        if task.family == "path_planning":
            opening_row = task.metadata.get("opening_row")
            if isinstance(opening_row, (int, np.integer)):
                task.metadata["opening_row"] = int(np.clip(int(opening_row) + dy, 1, self.config.grid_height - 2))
        task.metadata["translation"] = {"dy": dy, "dx": dx}

    def _translate_grid(self, grid: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
        translated = np.zeros_like(grid)

        src_y_start = max(0, -dy)
        src_y_end = min(grid.shape[0], grid.shape[0] - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(grid.shape[1], grid.shape[1] - dx)

        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        if src_y_start < src_y_end and src_x_start < src_x_end:
            translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = grid[src_y_start:src_y_end, src_x_start:src_x_end]

        return translated

    def _shift_target_zone(self, zone: TargetZone, *, dy: int, dx: int) -> TargetZone:
        north, east, down = zone.center_ned
        shifted_north = north - (dy * self.cell_size_m)
        shifted_east = east + (dx * self.cell_size_m)
        return TargetZone(center_ned=(shifted_north, shifted_east, down), radius_m=zone.radius_m)

    def _build_scene_descriptor(self, task: BenchmarkTask, rng: np.random.Generator) -> dict[str, Any]:
        """Creates a structured specification for NVIDIA Replicator."""
        entities = []
        grid = task.input_grid.values
        
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                val = grid[y, x]
                if val == 0: continue
                
                pos = self._grid_to_ned(y, x)
                entities.append({
                    "prim_type": "cube" if val != 3 else "sphere",
                    "position": pos,
                    "scale": (self.cell_size_m * 0.9, self.cell_size_m * 0.9, self.cell_size_m * 0.9),
                    "color_id": int(val),
                    "material": self.material_map.get(int(val), "plastic"),
                    "semantics": {"class": "arc_object", "id": f"obj_{y}_{x}"}
                })

        return {
            "scene_id": task.task_id,
            "entities": entities,
            "replicator_randomization": {
                "light_intensity_range": [800, 2500],
                "texture_jitter": True,
                "camera_noise_level": float(rng.uniform(0.01, 0.05)),
                "camera_pose_jitter": (0.05, 0.05, 0.05),  # Meters
                "material_shuffle": bool(rng.integers(0, 2)),
            }
        }

    def _generate_reasoning_trace(self, task: BenchmarkTask, rng: np.random.Generator) -> str:
        family = task.family
        if family == "symmetry":
            variants = [
                "Reasoning: Symmetry task. The input grid shows a pattern that must be reflected. I will identify the axis of symmetry and predict the drone movement to reach the reflected target.",
                "Reasoning: Symmetry task. I compare mirrored regions of the grid, infer the reflection rule, and align the drone decision with that mirrored structure.",
            ]
            return str(rng.choice(variants))
        if family == "counting":
            color_id = task.metadata.get("count_color", "?")
            count = task.metadata.get("count", "?")
            variants = [
                f"Reasoning: Counting task. I must identify all pixels of target color {color_id}, count them, and use that count to determine the drone destination.",
                f"Reasoning: Counting task. I scan the grid for color {color_id}, compute the total ({count}), and map that density to the correct action.",
            ]
            return str(rng.choice(variants))
        if family == "composition":
            variants = [
                "Reasoning: Composition task. I fuse overlapping spatial structures and track how the merged shape changes the correct movement choice.",
                "Reasoning: Composition task. I analyze the rotated and merged regions, then choose the action that matches the composed pattern.",
            ]
            return str(rng.choice(variants))
        if family == "path_planning":
            opening_row = task.metadata.get("opening_row", "?")
            variants = [
                f"Reasoning: Path planning task. I locate the obstacle opening near row {opening_row} and choose the maneuver that threads the drone through the gap.",
                f"Reasoning: Path planning task. I trace the free corridor around the central wall, verify the opening at row {opening_row}, and align the action with that route.",
            ]
            return str(rng.choice(variants))
        return f"Reasoning: {family} task. Analyzing spatial relations between objects."

    def _make_symmetry_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = self._random_grid(rng)
        target = np.fliplr(grid)
        return self._task(
            index=index, family="symmetry", input_grid=grid, target_grid=target,
            action=DroneAction((0.0, 0.3, 0.0), 0.0, 0.95),
            target_zone=TargetZone((0.0, 1.2, -1.5), 0.45),
            target_entity_name=default_target_entity_name("symmetry"),
            metadata={"transform": "mirror_x"}
        )

    def _make_counting_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = self._random_grid(rng)
        color = int(rng.integers(1, 9))
        count = int(np.sum(grid == color))
        target = np.full_like(grid, count % 10)
        return self._task(
            index=index, family="counting", input_grid=grid, target_grid=target,
            action=DroneAction((0.3, 0.0, 0.0), 0.0, 0.9),
            target_zone=TargetZone((1.0, 0.0, -1.2), 0.4),
            target_entity_name=default_target_entity_name("counting"),
            metadata={"count_color": color, "count": count}
        )

    def _make_composition_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = self._random_grid(rng)
        rotated = np.rot90(grid)
        target = np.where(rotated > 0, rotated, grid)
        return self._task(
            index=index, family="composition", input_grid=grid, target_grid=target,
            action=DroneAction((0.0, 0.0, 0.3), 0.0, 0.88),
            target_zone=TargetZone((0.0, 0.0, -2.0), 0.35),
            target_entity_name=default_target_entity_name("composition"),
            metadata={"transform": "rotate_merge"}
        )

    def _make_path_planning_task(self, rng: np.random.Generator, index: int) -> BenchmarkTask:
        grid = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.int64)
        opening = int(rng.integers(1, self.config.grid_height - 1))
        grid[:, self.config.grid_width // 2] = 2
        grid[opening - 1:opening + 2, self.config.grid_width // 2] = 0
        target = grid.copy()
        target[opening, -1] = 4
        return self._task(
            index=index, family="path_planning", input_grid=grid, target_grid=target,
            action=DroneAction((0.0, 0.0, 0.0), 0.25, 0.9),
            target_zone=TargetZone((1.8, 0.0, -1.5), 0.4),
            target_entity_name=default_target_entity_name("path_planning"),
            metadata={"opening_row": opening}
        )

    def _task(self, index: int, family: str, input_grid: np.ndarray, target_grid: np.ndarray,
              action: DroneAction, target_zone: TargetZone, target_entity_name: str | None,
              metadata: dict[str, Any]) -> BenchmarkTask:
        return BenchmarkTask(
            task_id=f"{family}-{index:04d}", family=family,
            input_grid=ArcGrid(input_grid.astype(np.int64)),
            target_grid=ArcGrid(target_grid.astype(np.int64)),
            target_action=action, target_zone=target_zone,
            target_entity_name=target_entity_name, metadata=metadata
        )

    def _random_grid(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(low=0, high=10, size=(self.config.grid_height, self.config.grid_width), dtype=np.int64)
