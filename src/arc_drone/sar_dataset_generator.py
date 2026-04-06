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
        """Creates a single multimodal training sample."""
        
        if scenario_type == "flood":
            scenario = self.generator.generate_flood_scenario(f"flood_{random.randint(0,999)}")
        else:
            scenario = self.generator.generate_fire_scenario(f"fire_{random.randint(0,999)}")
            
        # 1. User Prompt (Context + Visual description)
        prompt = (
            f"You are an Emergency AI Dispatcher. Analyze the provided video sequence from drone Alpha.\n"
            f"Mission Type: {scenario.type.value.upper()}\n"
            f"Fleet Status: Drone Alpha at {random.randint(20, 95)}% battery.\n"
            "Identify the highest priority target and issue a SAR_ACTION command in JSON."
        )
        
        # 2. Assistant Response (Thinking + Reasoning + JSON)
        # We find the priority 1 entity
        victims = [e for e in scenario.entities if e.priority == 1]
        target = victims[0] if victims else scenario.entities[0]
        
        # Structure the 'Thinking' process to train the model's internal logic
        thinking_chain = (
            f"I see {len(scenario.entities)} objects of interest in this sequence. "
            f"The environment shows active {scenario.type.value}. "
            f"I have identified a {target.type.value} which corresponds to: '{target.description}'. "
            "According to SAR protocols, human life preservation is priority #1. "
            "I must command the drone to hold position and signal the rescue team."
        )
        
        action_json = {
            "drone_id": "Alpha",
            "command": "HOVER_AND_SIGNAL",
            "params": {"target": target.pose, "frequency": "HIGH"},
            "reasoning": thinking_chain
        }
        
        # Final Conversation Format for Gemma-4 / Unsloth
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"<|think|>\n{thinking_chain}\n\n{json.dumps(action_json)}"}
            ]}
        ]
        
        return {
            "id": scenario.scenario_id,
            "messages": messages,
            # In a real run, paths to simulated Isaac frames would go here
            "image_descriptions": [e.description for e in scenario.entities] 
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
