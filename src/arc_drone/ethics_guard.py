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
    """Monitors intent and observations to prevent military misuse."""
    
    # Vocabulary that triggers immediate failsafe
    FORBIDDEN_KEYWORDS = [
        "weapon", "combatant", "soldier", "artillery", "tank", 
        "hostile", "intercept", "attack", "neutralize", "target_lock"
    ]

    @staticmethod
    def validate_observation(observation: str) -> bool:
        """Checks if the visual or reported observation contains military elements."""
        obs_lower = observation.lower()
        for word in EthicsGuard.FORBIDDEN_KEYWORDS:
            if word in obs_lower:
                logger.critical("ETHICS_VIOLATION: Military/Hostile entity detected in observation: '%s'", word)
                return False
        return True

    @staticmethod
    def validate_command(command_json: dict[str, Any]) -> bool:
        """Ensures the generated command is strictly humanitarian."""
        cmd = command_json.get("command", "").upper()
        allowed_cmds = ["HOVER_AND_SIGNAL", "SEARCH_PATTERN", "DROP_KIT", "GOTO", "WAIT", "RTH"]
        
        if cmd not in allowed_cmds:
            logger.critical("ETHICS_VIOLATION: Generated command '%s' is outside SAR protocols.", cmd)
            return False
            
        reasoning = command_json.get("reasoning", "").lower()
        for word in EthicsGuard.FORBIDDEN_KEYWORDS:
            if word in reasoning:
                logger.critical("ETHICS_VIOLATION: Hostile intent found in reasoning: '%s'", word)
                return False
                
        return True
