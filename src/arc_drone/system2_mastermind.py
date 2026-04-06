"""
System 2 Mastermind: Strategic orchestration and OOD reasoning using Gemma-4.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Optional
from .system1_pilot import Waypoint

logger = logging.getLogger(__name__)

class System2Mastermind:
    """Strategic Cloud/Heavy-Edge Mastermind."""
    
    def __init__(self, model_id: str = "unsloth/gemma-4-e4b-it", use_real_model: bool = False):
        self.model_id = model_id
        self.use_real_model = use_real_model
        self.model: Any = None
        self.processor: Any = None
        
        if use_real_model:
            self._load_model()

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
            # Look for the first '{' and last '}'
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {}
        except Exception as e:
            logger.error("Failed to parse Mastermind JSON: %s | Text: %s", e, text)
            return {}

    def _generate_prompt(self, swarm_state: list[dict[str, Any]], ood_event: Optional[dict[str, Any]] = None) -> str:
        prompt = "You are the Strategic Mastermind for an autonomous drone fleet.\n"
        prompt += f"FLEET_STATUS: {json.dumps(swarm_state)}\n"
        
        if ood_event:
            prompt += f"\nANOMALY DETECTED by {ood_event['drone_id']}: {ood_event['observation']}\n"
            prompt += "Identify the object and provide a strategic detour.\n"
        else:
            prompt += "\nGOAL: Optimize fleet positions for maximum coverage.\n"
            
        prompt += "\nRESPONSE_FORMAT: Output ONLY a JSON object: {\"drone_id\": \"...\", \"command\": \"GOTO\", \"target\": [x, y, z], \"reasoning\": \"...\"}"
        return prompt

    def reason(self, swarm_state: list[dict[str, Any]], ood_event: Optional[dict[str, Any]] = None, image: Optional[Any] = None) -> dict[str, Any]:
        """Strategic reasoning step."""
        if not self.use_real_model:
            # Simulated Logic (Fallback)
            return self._simulated_reason(swarm_state, ood_event)

        # Real Inference Logic
        prompt = self._generate_prompt(swarm_state, ood_event)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if image:
            messages[0]["content"].insert(0, {"type": "image"})

        # Using standard chat template logic
        formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to("cuda")
        
        import torch
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        full_response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return self._extract_json(full_response)

    def _simulated_reason(self, swarm_state: list[dict[str, Any]], ood_event: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Simulated reasoning for development without GPU."""
        if ood_event:
            return {
                "drone_id": ood_event["drone_id"],
                "command": "GOTO",
                "target": [ood_event["pose"][0], ood_event["pose"][1], ood_event["pose"][2] + 2.0],
                "reasoning": "Detected an unknown obstacle. Increasing altitude to bypass."
            }
        return {
            "drone_id": swarm_state[0]["drone_id"],
            "command": "GOTO",
            "target": [100.0, 50.0, 15.0],
            "reasoning": "Standard patrol route update."
        }

    def execute_strategy(self, pilot: Any, strategy: dict[str, Any]) -> None:
        if strategy.get("command") == "GOTO":
            t = strategy["target"]
            wp = Waypoint(x=t[0], y=t[1], z=t[2], description=strategy.get("reasoning", ""))
            pilot.set_waypoint(wp)
