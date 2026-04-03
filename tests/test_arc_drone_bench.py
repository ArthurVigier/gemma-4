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
