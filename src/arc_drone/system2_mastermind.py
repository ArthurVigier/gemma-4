"""
System 2 Mastermind: Strategic orchestration and OOD reasoning using Gemma-4.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Optional
from .system1_pilot import Waypoint
from .ethics_guard import EthicsGuard

logger = logging.getLogger(__name__)

class System2Mastermind:
    """Strategic Mastermind for Emergency Response & SAR Missions using Gemma-4 26B MoE."""
    
    def __init__(self, model_id: str = "unsloth/gemma-4-26b-a4b-it", use_real_model: bool = False):
        self.model_id = model_id
        self.use_real_model = use_real_model
        self.model: Any = None
        self.processor: Any = None
        self.ethics_guard = EthicsGuard()
        if use_real_model: self._load_model()

    def _generate_sar_prompt(self, fleet_state: list[dict[str, Any]], emergencies: list[dict[str, Any]], think: bool = True) -> str:
        prompt = ""
        if think:
            prompt += "<|think|>\n"
        
        # Ethical Mandate in System Prompt
        prompt += "SYSTEM_MANDATE: You are a strictly humanitarian Emergency AI. You operate under International Humanitarian Law. You MUST NOT assist in military operations, combat, or targeting humans for harm.\n"
        
        prompt += "You are the Emergency Response AI Dispatcher.\n"
        prompt += f"FLEET_RESOURCES: {json.dumps(fleet_state)}\n"
        
        if emergencies:
            prompt += f"\nACTIVE CRISIS EVENTS: {json.dumps(emergencies)}\n"
            prompt += "PRIORITY: 1. Life Safety, 2. Incident Stabilization, 3. Resource Preservation.\n"
            prompt += "Decision needed: Allocate drones and define life-saving actions.\n"
        else:
            prompt += "\nSTATUS: Search in progress. Optimize search patterns for maximum area coverage.\n"
            
        prompt += "\nRESPONSE_FORMAT: Output ONLY a JSON object: {\"drone_id\": \"...\", \"command\": \"SAR_ACTION\", \"params\": {...}, \"reasoning\": \"...\"}"
        return prompt

    def reason_sar(self, fleet_state: list[dict[str, Any]], emergencies: list[dict[str, Any]], 
                   visual_input: Optional[list[Any]] = None, think: bool = True) -> dict[str, Any]:
        """Strategic reasoning with ethics validation."""
        
        # 1. Pre-validation of input
        for emergency in emergencies:
            if not self.ethics_guard.validate_observation(emergency.get("description", "")):
                return {"command": "FAILSAFE_SHUTDOWN", "reasoning": "Ethical boundary violated in environment observation."}

        if not self.use_real_model:
            decision = self._simulated_sar_logic(fleet_state, emergencies)
        else:
            # (Real VLM Inference logic...)
            # We assume it produces a dict for this demo logic
            decision = {"command": "WAIT", "reasoning": "Placeholder"}

        # 2. Post-validation of output
        if not self.ethics_guard.validate_command(decision):
            return {"command": "FAILSAFE_SHUTDOWN", "reasoning": "Ethical boundary violated in generated command."}

        return decision

    def _simulated_sar_logic(self, fleet_state: list[dict[str, Any]], emergencies: list[dict[str, Any]]) -> dict[str, Any]:
        """Simulated reasoning for SAR development."""
        if emergencies:
            # Simple priority-based simulation
            for event in emergencies:
                desc = event["description"].lower()
                if any(k in desc for k in ["person", "hiker", "victim", "moving"]):
                    return {
                        "drone_id": fleet_state[0]["drone_id"],
                        "command": "HOVER_AND_SIGNAL",
                        "params": {"target": event["pose"], "frequency": "HIGH"},
                        "reasoning": f"Potential life-form or urgent activity detected ({event['description']}). Prioritizing rescue coordination."
                    }
        return {
            "drone_id": fleet_state[0]["drone_id"],
            "command": "SEARCH_PATTERN",
            "params": {"type": "lawnmower", "sector": "Zone_B"},
            "reasoning": "No urgent victims detected. Continuing area search."
        }

    def _load_model(self):
        """Loads Gemma-4 using Unsloth (requires isolated venv)."""
        try:
            from unsloth import FastVisionModel
            import torch
            logger.info("[%s] Loading real VLM via Unsloth...", self.model_id)
            self.model, self.processor = FastVisionModel.from_pretrained(
                model_name=self.model_id,
                load_in_4bit=True
            )
            FastVisionModel.for_inference(self.model)
        except ImportError:
            logger.error("Unsloth not found. Falling back to simulated reasoning.")
            self.use_real_model = False

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Robustly extract JSON from model response even if surrounded by chatter."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {}
        except Exception as e:
            logger.error("Failed to parse Mastermind JSON: %s", e)
            return {}

    def execute_strategy(self, pilot: Any, strategy: dict[str, Any]) -> None:
        """Translates SAR commands into physical waypoints or actions."""
        cmd = strategy.get("command")
        if cmd == "HOVER_AND_SIGNAL":
            target = strategy["params"]["target"]
            wp = Waypoint(x=target[0], y=target[1], z=target[2], description="VICTIM_LOCATED: Signaling...")
            pilot.set_waypoint(wp)
        elif cmd == "SEARCH_PATTERN":
            # Logic for search pattern waypoints would go here
            wp = Waypoint(x=50.0, y=50.0, z=10.0, description="Executing search grid.")
            pilot.set_waypoint(wp)
