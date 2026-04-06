"""
Crisis VLM Dataset Generator: Produces high-fidelity SAR training data.

Generates conversation pairs (User/Assistant) for Gemma-4 26B MoE fine-tuning,
focusing on multi-frame video reasoning, triage logic, and Thinking mode.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, List
from .sar_scenario import SARScenarioGenerator, EmergencyType, EntityType

class CrisisVLMDatasetGenerator:
    """Generates a synthetic dataset for SAR strategic reasoning."""
    
    def __init__(self, output_path: str = "data/sar_training_vlm.jsonl"):
        self.output_path = Path(output_path)
        self.generator = SARScenarioGenerator()

    def generate_training_sample(self, scenario_type: str) -> dict[str, Any]:
        """Creates a sample with structured Reasoning Traces (Chain of Thought)."""
        
        if scenario_type == "flood":
            scenario = self.generator.generate_flood_scenario(f"flood_{random.randint(0,999)}")
        else:
            scenario = self.generator.generate_fire_scenario(f"fire_{random.randint(0,999)}")
            
        # User Prompt
        prompt = (
            f"EMERGENCY DISPATCH PROTOCOL: Analyze drone Alpha video feed.\n"
            f"SCENARIO: {scenario.type.value.upper()}\n"
            f"FLEET: Alpha (Battery: {random.randint(20, 95)}%).\n"
            "Evaluate entities and execute SAR_ACTION."
        )
        
        # Identify targets
        entities = scenario.entities
        primary_target = min(entities, key=lambda e: e.priority)
        
        # Structured Reasoning Trace
        thinking_chain = (
            f"[OBSERVATION] The video sequence reveals {len(entities)} distinct entities in an active {scenario.type.value} environment. "
            f"Found: {', '.join([e.description for e in entities])}. "
            "[PRIORITY ANALYSIS] Comparing risks. "
        )
        
        if len(entities) > 1:
            thinking_chain += (
                f"Entity '{primary_target.id}' (Priority {primary_target.priority}) presents a more critical life-safety risk "
                f"than other observed hazards. Battery levels are sufficient for immediate intervention. "
            )
        
        thinking_chain += (
            f"[DECISION] Prioritizing {primary_target.type.value}. "
            "Reason: Human life preservation mandate (Protocol IAMSAR). "
            "Action: Deploy signal and hold position for rescue team."
        )
        
        action_json = {
            "drone_id": "Alpha",
            "command": "HOVER_AND_SIGNAL",
            "params": {"target": primary_target.pose, "frequency": "HIGH"},
            "reasoning": thinking_chain
        }
        
        # Conversation Format with <|think|> for Gemma-4 26B MoE
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"<|think|>\n{thinking_chain}\n\n{json.dumps(action_json)}"}
            ]}
        ]
        
        return {
            "id": scenario.scenario_id,
            "messages": messages,
            "target_priority": primary_target.priority
        }

    def build_dataset(self, count: int = 1000):
        """Generates a complete JSONL dataset."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            for i in range(count):
                stype = random.choice(["flood", "fire"])
                sample = self.generate_training_sample(stype)
                f.write(json.dumps(sample) + "\n")
                
        return self.output_path
