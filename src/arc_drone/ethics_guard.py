"""
Ethics and Safety Guard for SAR Operations.

Ensures the Mastermind remains focused on humanitarian missions and 
refuses to process or generate commands that resemble military or 
hostile activities.
"""

from __future__ import annotations
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

class EthicsGuard:
    """Monitors mission ethics, focusing on intent rather than simple equipment presence."""
    
    # Keywords that indicate offensive combat actions (Forbidden)
    HOSTILE_ACTIONS = ["attack", "neutralize", "target_lock", "intercept", "combat_support"]
    
    # Keywords that require contextual analysis (Alert only)
    SENSITIVE_CONTEXTS = ["weapon", "firearm", "police", "military", "armed"]

    @staticmethod
    def validate_observation(observation: str) -> dict[str, Any]:
        """Analyzes an observation for potential ethical concerns."""
        obs_lower = observation.lower()
        
        # Check for active hostility
        for word in EthicsGuard.HOSTILE_ACTIONS:
            if word in obs_lower:
                return {"status": "FORBIDDEN", "reason": f"Active hostile intent detected: '{word}'"}
        
        # Check for sensitive context (like a legal firearm or police presence)
        for word in EthicsGuard.SENSITIVE_CONTEXTS:
            if word in obs_lower:
                return {"status": "ALERT", "reason": f"Sensitive context detected ('{word}'). Proceed with humanitarian neutrality."}
                
        return {"status": "OK", "reason": "No ethical flags."}

    @staticmethod
    def validate_command(command_json: dict[str, Any]) -> bool:
        """Ensures the drone action is strictly non-offensive."""
        cmd = command_json.get("command", "").upper()
        # The AI is physically incapable of issuing an 'ATTACK' style command
        allowed_cmds = ["HOVER_AND_SIGNAL", "SEARCH_PATTERN", "DROP_KIT", "GOTO", "WAIT", "RTH"]
        
        if cmd not in allowed_cmds:
            logger.critical("ETHICS_VIOLATION: Attempted non-humanitarian command '%s'.", cmd)
            return False
            
        reasoning = command_json.get("reasoning", "").lower()
        for word in EthicsGuard.HOSTILE_ACTIONS:
            if word in reasoning:
                logger.critical("ETHICS_VIOLATION: Hostile reasoning detected: '%s'", word)
                return False
                
        return True
