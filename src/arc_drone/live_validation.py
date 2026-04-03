"""Validation helpers for the live Gazebo mission world and marker topology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig
from .mission_targets import default_mission_target_subscriptions


@dataclass(frozen=True, slots=True)
class MissionWorldValidationReport:
    """Structured report for local mission-world validation."""

    world_path: str
    benchmark_target_entities: tuple[str, ...]
    world_model_names: tuple[str, ...]
    missing_entities: tuple[str, ...]
    missing_topics: tuple[str, ...]
    ok: bool


def benchmark_target_entities(config: BenchmarkConfig | None = None) -> tuple[str, ...]:
    """Returns the unique mission target entities required by the benchmark."""

    bench = ARCDroneBench(config or BenchmarkConfig(task_count=200))
    entities = sorted({task.target_entity_name for task in bench.generate_tasks() if task.target_entity_name is not None})
    return tuple(entities)


def world_model_names(world_path: str | Path) -> tuple[str, ...]:
    """Extracts model names from explicit `<model>` and `<include>` SDF entries."""

    root = ElementTree.fromstring(Path(world_path).read_text(encoding="utf-8"))
    names: set[str] = {
        element.attrib["name"]
        for element in root.iter()
        if element.tag == "model" and "name" in element.attrib
    }
    for include in root.iter("include"):
        uri = include.findtext("uri", default="")
        if uri.startswith("model://"):
            names.add(uri.removeprefix("model://"))
    return tuple(names)


def validate_mission_world(world_path: str | Path) -> MissionWorldValidationReport:
    """Validates that the Gazebo world provides all benchmark mission markers."""

    required_entities = benchmark_target_entities()
    present_models = world_model_names(world_path)
    present_model_set = set(present_models)
    missing_entities = tuple(entity for entity in required_entities if entity not in present_model_set)

    subscription_topics = {subscription.entity_name: subscription.topic_name for subscription in default_mission_target_subscriptions()}
    missing_topics = tuple(entity for entity in required_entities if entity not in subscription_topics)

    return MissionWorldValidationReport(
        world_path=Path(world_path).as_posix(),
        benchmark_target_entities=required_entities,
        world_model_names=present_models,
        missing_entities=missing_entities,
        missing_topics=missing_topics,
        ok=not missing_entities and not missing_topics,
    )


def validation_summary(report: MissionWorldValidationReport) -> str:
    """Formats a concise text summary for CLI validation output."""

    status = "OK" if report.ok else "FAILED"
    return (
        f"Mission world validation {status}: world={report.world_path} "
        f"targets={len(report.benchmark_target_entities)} "
        f"models={len(report.world_model_names)} "
        f"missing_entities={list(report.missing_entities)} "
        f"missing_topics={list(report.missing_topics)}"
    )
