import json
import sys
import types

import numpy as np
from PIL import Image

from arc_drone.arc_drone_bench import ARCDroneBench
from arc_drone.config import BenchmarkConfig


def test_benchmark_generates_minimum_200_tasks() -> None:
    bench = ARCDroneBench(BenchmarkConfig(task_count=200))
    tasks = bench.generate_tasks()

    assert len(tasks) == 200
    assert {task.family for task in tasks} == {"symmetry", "counting", "composition", "path_planning"}


def test_path_planning_task_contains_exit_marker() -> None:
    bench = ARCDroneBench(BenchmarkConfig(task_count=4, seed=11))
    task = next(item for item in bench.generate_tasks() if item.family == "path_planning")

    assert 4 in task.target_grid.values
    assert task.target_entity_name == "arc_marker_path_planning"


def test_benchmark_augmented_tasks_include_provenance_metadata() -> None:
    bench = ARCDroneBench(BenchmarkConfig(task_count=4, seed=7))
    tasks = bench.generate_tasks(augment=True)

    assert all(task.metadata["dataset_version"] == "arc_drone_v3_hybrid_auto" for task in tasks)
    assert all("source_type" in task.metadata for task in tasks)
    assert all("augmentations" in task.metadata for task in tasks)
    assert all("isaac_scene" in task.metadata for task in tasks)
    assert all("reasoning_trace" in task.metadata for task in tasks)


def test_benchmark_auto_discovers_real_manifest(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "arc_drone_real_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": "real-001",
                "family": "real_world",
                "input_grid": [[1, 0], [0, 2]],
                "target_grid": [[1, 0], [0, 2]],
                "action_index": 0,
                "halt_step": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARC_DRONE_REAL_DATA_PATH", str(manifest_path))

    bench = ARCDroneBench(BenchmarkConfig(task_count=1, real_data_ratio=1.0))
    tasks = bench.generate_tasks()

    assert len(tasks) == 1
    assert tasks[0].task_id == "real-001"
    assert tasks[0].metadata["source_type"] == "real"
    assert tasks[0].metadata["dataset_version"] == "arc_drone_v3_hybrid_auto"
    assert "isaac_scene" in tasks[0].metadata
    assert "reasoning_trace" in tasks[0].metadata


def test_benchmark_loads_real_hf_dataset_via_preset(monkeypatch) -> None:
    class _FakeStream:
        def __init__(self, rows):
            self._rows = rows

        def shuffle(self, *, buffer_size, seed):
            return self

        def __iter__(self):
            return iter(self._rows)

    fake_image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    rows = [
        {
            "width": 32,
            "height": 32,
            "objects": {"bbox": [[4, 4, 10, 12]], "category": [2], "area": [120], "id": [1]},
            "image": fake_image,
            "image_id": 7,
        }
    ]

    def _fake_load_dataset(name, *, split, streaming):
        assert name == "pathikg/drone-detection-dataset"
        assert split == "train"
        assert streaming is True
        return _FakeStream(rows)

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = _fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    bench = ARCDroneBench(BenchmarkConfig(task_count=1, real_dataset="drone_detection", real_data_ratio=1.0))
    tasks = bench.generate_tasks()

    assert len(tasks) == 1
    assert tasks[0].metadata["source_type"] == "real_hf"
    assert tasks[0].metadata["hf_dataset_id"] == "pathikg/drone-detection-dataset"
    assert "pil_image" in tasks[0].metadata
    assert "isaac_scene" in tasks[0].metadata
    assert int(tasks[0].input_grid.values.max()) > 0
