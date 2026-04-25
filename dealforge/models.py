from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    PROPOSE_CONTRACT = "propose_contract"
    MODIFY_CLAUSE = "modify_clause"
    ACCEPT_DEAL = "accept_deal"
    REJECT_DEAL = "reject_deal"
    DELAY_NEGOTIATION = "delay_negotiation"
    ASK_QUESTION = "ask_question"


@dataclass
class Action:
    action_type: ActionType
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Clause:
    clause_id: str
    clause_type: str
    content: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ContractVersion:
    version_id: str
    parent_version_id: Optional[str]
    clauses: Dict[str, Clause]
    proposer: str
    step_created: int
    status: str = "draft"


@dataclass
class Event:
    event_id: str
    step_scheduled: int
    event_type: str
    payload: Dict[str, Any]


@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
