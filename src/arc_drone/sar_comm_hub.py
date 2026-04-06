"""
SAR Communication Hub: Natural language interface for human emergency services.

Translates Mastermind strategic decisions into clear, human-readable situational 
reports (SITREPs) for rescue teams on the ground.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, List

logger = logging.getLogger(__name__)

class SARCommunicationHub:
    """Interface for human rescue teams to monitor the autonomous swarm."""
    
    def __init__(self):
        self.message_history: List[dict[str, Any]] = []

    def broadcast_sitrep(self, drone_id: str, action: dict[str, Any]):
        """Generates and logs a SITREP (Situational Report)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Extract strategic reasoning for transparency
        reasoning = action.get("reasoning", "No detailed reasoning provided.")
        command = action.get("command", "WAIT")
        params = action.get("params", {})
        
        # Format for human dispatchers
        msg = f"[{timestamp}] UNIT: {drone_id} | STATUS: {command} | INTENT: {reasoning}"
        
        if "target" in params:
            msg += f" | LOCATION: {params['target']}"
            
        logger.info("BROADCAST [Rescue Channel]: %s", msg)
        
        self.message_history.append({
            "timestamp": timestamp,
            "drone_id": drone_id,
            "human_readable": msg,
            "raw_action": action
        })

    def get_mission_summary(self) -> str:
        """Returns a summary of all strategic interventions for the current shift."""
        summary = "--- SAR MISSION LOG SUMMARY ---\n"
        for entry in self.message_history:
            summary += f"{entry['human_readable']}\n"
        return summary

    def request_human_clarification(self, drone_id: str, observation: str):
        """Used when the Mastermind's uncertainty is too high (OOD fallback)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] !!! URGENT CLARIFICATION NEEDED FROM {drone_id} !!!"
        msg += f"\nOBSERVATION: {observation}"
        msg += "\nPlease confirm if this is a victim or hazard."
        
        logger.warning("ALERT [Human Dispatcher Required]: %s", msg)
        return msg
