from pathlib import Path

from arc_drone.live_validation import (
    benchmark_target_entities,
    validate_mission_world,
    validation_summary,
    world_model_names,
)


def test_benchmark_target_entities_cover_all_marker_models() -> None:
    entities = benchmark_target_entities()

    assert entities == (
        "arc_marker_composition",
        "arc_marker_counting",
        "arc_marker_path_planning",
        "arc_marker_symmetry",
    )


def test_world_model_names_extract_marker_models() -> None:
    world_path = Path("assets/gazebo/worlds/arc_drone_bench_mission.world")

    names = world_model_names(world_path)

    assert "ground_plane" in names


def test_validate_mission_world_reports_success_for_repo_world() -> None:
    world_path = Path("assets/gazebo/worlds/arc_drone_bench_mission.world")

    report = validate_mission_world(world_path)

    assert report.ok is True
    assert report.missing_entities == ()
    assert report.missing_topics == ()
    assert "OK" in validation_summary(report)
