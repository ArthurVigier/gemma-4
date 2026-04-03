from types import SimpleNamespace

from arc_drone.mission_targets import MissionTargetTracker, default_target_entity_name


def test_mission_target_tracker_converts_enu_to_ned_from_odometry() -> None:
    tracker = MissionTargetTracker(subscriptions=[])
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=250)),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=2.0, y=-4.0, z=1.5),
            )
        ),
    )

    state = tracker.ingest_odometry_message(
        entity_name="arc_marker_symmetry",
        odometry_message=message,
        source_topic="/arc_drone/mission_markers/arc_marker_symmetry/odometry",
    )

    assert state.timestamp_ns == 3_000_000_250
    assert state.position_enu == (2.0, -4.0, 1.5)
    assert state.position_ned == (-4.0, 2.0, -1.5)


def test_mission_target_tracker_reports_distance_and_reachability() -> None:
    tracker = MissionTargetTracker(subscriptions=[])
    tracker.update_target_position_ned(
        entity_name="arc_marker_path_planning",
        position_ned=(1.5, -0.5, -1.2),
    )

    distance_m = tracker.distance_to_target_ned(
        entity_name="arc_marker_path_planning",
        reference_position_ned=(1.6, -0.4, -1.2),
    )

    assert distance_m is not None
    assert round(distance_m, 3) == 0.141
    assert tracker.is_target_reached(
        entity_name="arc_marker_path_planning",
        reference_position_ned=(1.6, -0.4, -1.2),
        tolerance_m=0.2,
    )


def test_default_target_entity_name_covers_all_benchmark_families() -> None:
    assert default_target_entity_name("symmetry") == "arc_marker_symmetry"
    assert default_target_entity_name("counting") == "arc_marker_counting"
    assert default_target_entity_name("composition") == "arc_marker_composition"
    assert default_target_entity_name("path_planning") == "arc_marker_path_planning"
