"""
SAR Scenario Definitions: Data structures for simulating emergency crisis events.
Used to test the Mastermind's (System 2) strategic triage and orchestration.
"""

from __future__ import annotations
import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

class EmergencyType(Enum):
    FLOOD = "flood"
    FOREST_FIRE = "forest_fire"
    EARTHQUAKE = "earthquake"
    URBAN_SEARCH_AND_RESCUE = "usar"

class EntityType(Enum):
    VICTIM_SINGLE = "victim_single"
    VICTIM_GROUP = "victim_group"
    HAZARD_FIRE = "hazard_fire"
    HAZARD_WATER = "hazard_water"
    MEDICAL_KIT = "medical_kit"
    LANDING_ZONE = "landing_zone"

@dataclass
class SARObject:
    id: str
    type: EntityType
    pose: tuple[float, float, float]
    description: str
    priority: int  # 1 (Highest) to 5 (Lowest)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SARScenario:
    scenario_id: str
    type: EmergencyType
    bounds: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    entities: List[SARObject] = field(default_factory=list)
    initial_fleet_count: int = 2

    def to_json(self) -> str:
        d = asdict(self)
        # Convert enums to strings for JSON
        d["type"] = self.type.value
        for e in d["entities"]:
            e["type"] = e["type"].value
        return json.dumps(d, indent=2)

class SARScenarioGenerator:
    """Generates synthetic crisis scenarios for triage testing."""
    
    @staticmethod
    def generate_flood_scenario(scenario_id: str) -> SARScenario:
        entities = [
            SARObject(
                id="victim_0",
                type=EntityType.VICTIM_SINGLE,
                pose=(12.5, 45.0, 2.0),
                description="Person trapped on a submerged vehicle.",
                priority=1
            ),
            SARObject(
                id="victim_1",
                type=EntityType.VICTIM_GROUP,
                pose=(-30.0, 10.0, 5.0),
                description="Family of 4 on a residential roof.",
                priority=1,
                metadata={"victim_count": 4}
            ),
            SARObject(
                id="hazard_0",
                type=EntityType.HAZARD_WATER,
                pose=(0.0, 0.0, 0.0),
                description="Fast moving water current in Sector A.",
                priority=2
            )
        ]
        return SARScenario(
            scenario_id=scenario_id,
            type=EmergencyType.FLOOD,
            bounds=(-100, -100, 100, 100),
            entities=entities
        )

    @staticmethod
    def generate_fire_scenario(scenario_id: str) -> SARScenario:
        entities = [
            SARObject(
                id="fire_front",
                type=EntityType.HAZARD_FIRE,
                pose=(50.0, 20.0, 0.0),
                description="Advancing fire line near power station.",
                priority=2
            ),
            SARObject(
                id="victim_isolated",
                type=EntityType.VICTIM_SINGLE,
                pose=(55.0, 15.0, 0.0),
                description="Hiker trapped by smoke in a valley.",
                priority=1
            )
        ]
        return SARScenario(
            scenario_id=scenario_id,
            type=EmergencyType.FOREST_FIRE,
            bounds=(-200, -200, 200, 200),
            entities=entities
        )
