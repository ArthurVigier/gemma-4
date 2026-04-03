from arc_drone.gemma_layer_sweep import choose_layer_indices, serialize_task_for_teacher
from arc_drone.arc_drone_bench import ARCDroneBench
from arc_drone.config import BenchmarkConfig


def test_choose_layer_indices_from_fractions() -> None:
    layers = choose_layer_indices(total_layers=40, explicit_layers=(), fractions=(0.25, 0.5, 0.75, 0.9))

    assert layers == [10, 20, 29, 35]


def test_serialize_task_for_teacher_contains_family_and_grid() -> None:
    task = ARCDroneBench(BenchmarkConfig(task_count=1)).generate_tasks()[0]

    prompt = serialize_task_for_teacher(task)

    assert task.family in prompt
    assert "Input grid:" in prompt
