from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class DealForgeAction(Action):
    """Primary environment action for negotiation turns."""

    type: Literal["dealforge_action"] = "dealforge_action"
    action_name: Literal[
        "propose_contract",
        "modify_clause",
        "accept_deal",
        "reject_deal",
        "delay_negotiation",
        "ask_question",
    ] = Field(description="Negotiation tool/action selected by the learning agent")
    payload: Dict[str, Any] = Field(default_factory=dict)


class DealForgeObservation(Observation):
    """Observation returned at each step (POMDP visible slice only)."""

    visible_state: Dict[str, Any] = Field(default_factory=dict)
    event_log: List[Dict[str, Any]] = Field(default_factory=list)


class DealForgeState(State):
    """Internal full state for debugging/evaluation (includes hidden state)."""

    dynamic_state: Dict[str, Any] = Field(default_factory=dict)
