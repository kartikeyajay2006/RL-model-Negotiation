from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .agents import CompetitorAgent, LegalAgent, RiskAgent, SellerAgent
from .config import DealForgeConfig
from .models import Action, ActionType, Clause, ContractVersion, Event, StepResult


class DealForgeEnv:
    """Config-driven partially observable multi-agent negotiation environment."""

    def __init__(self, config: DealForgeConfig | Dict[str, Any], seed: Optional[int] = None):
        self.config = config if isinstance(config, DealForgeConfig) else DealForgeConfig(config)
        self.rng = random.Random(seed)
        self.seed = seed

        self.step_count = 0
        self.horizon = 0
        self.terminated = False
        self.truncated = False

        self.hidden_state: Dict[str, Any] = {}
        self.public_state: Dict[str, Any] = {}
        self.contract_versions: Dict[str, ContractVersion] = {}
        self.current_contract_id: Optional[str] = None
        self.events: List[Event] = []
        self.history: List[Dict[str, Any]] = []

        self._init_internal_agents()

    def _init_internal_agents(self) -> None:
        agents_cfg = self.config.section("agents")
        self.seller = SellerAgent("seller", agents_cfg.get("seller", {}), self.rng)
        self.legal = LegalAgent("legal", agents_cfg.get("legal", {}), self.rng)
        self.risk = RiskAgent("risk", agents_cfg.get("risk", {}), self.rng)
        self.competitor = CompetitorAgent("competitor", agents_cfg.get("competitor", {}), self.rng)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
            self.seed = seed

        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.history = []
        self.events = []
        self.contract_versions = {}

        scenario = self._sample_scenario()
        self.horizon = scenario["horizon"]
        self.hidden_state = scenario["hidden_state"]
        self.public_state = scenario["public_state"]

        initial_contract = self._new_contract_version(
            parent_id=None,
            proposer="system",
            clauses=scenario["initial_clauses"],
        )
        self.current_contract_id = initial_contract.version_id
        self.contract_versions[self.current_contract_id] = initial_contract

        obs = self._build_observation()
        info = {"seed": self.seed, "horizon": self.horizon}
        return obs, info

    def step(self, action: Action | Dict[str, Any]) -> StepResult:
        if self.terminated or self.truncated:
            raise RuntimeError("Episode already finished. Call reset().")

        action_obj = self._normalize_action(action)
        action_ok, validation_msg = self._validate_action(action_obj)

        transition_notes = []
        if not action_ok:
            transition_notes.append({"type": "invalid_action", "message": validation_msg})
        else:
            transition_notes.extend(self._apply_buyer_action(action_obj))

        seller_out = self.seller.respond(self.public_state, self.hidden_state)
        legal_out = self.legal.evaluate(self._contract_summary())
        risk_out = self.risk.evaluate(self.public_state, self.hidden_state)
        competitor_events = self.competitor.maybe_inject(self.public_state)

        for ce in competitor_events:
            self.events.append(
                Event(
                    event_id=self._id("evt"),
                    step_scheduled=self.step_count + 1,
                    event_type=ce["event_type"],
                    payload=ce,
                )
            )

        fired_events = self._fire_due_events()
        self._update_public_state(action_obj, seller_out, legal_out, risk_out, fired_events)

        reward, reward_breakdown = self._compute_reward(
            action_valid=action_ok,
            action=action_obj,
            legal_out=legal_out,
            risk_out=risk_out,
            seller_out=seller_out,
            fired_events=fired_events,
        )

        self.step_count += 1
        self.public_state["step"] = self.step_count

        self.terminated = self._is_terminated(action_obj, legal_out, risk_out)
        self.truncated = not self.terminated and self.step_count >= self.horizon

        obs = self._build_observation()
        info = {
            "reward_breakdown": reward_breakdown,
            "action_valid": action_ok,
            "transition_notes": transition_notes,
            "legal_out": legal_out,
            "risk_out": risk_out,
            "seller_out": seller_out,
            "fired_events": fired_events,
            "current_contract_id": self.current_contract_id,
        }

        self.history.append(
            {
                "step": self.step_count,
                "action": action_obj,
                "reward": reward,
                "info": info,
            }
        )

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=info,
        )

    def _sample_scenario(self) -> Dict[str, Any]:
        scenario_cfg = self.config.section("scenario")
        horizon_min, horizon_max = scenario_cfg.get("horizon_range", [10, 30])
        horizon = self.rng.randint(horizon_min, horizon_max)

        hidden_state = {
            "seller_reservation_utility": self.rng.uniform(*scenario_cfg.get("seller_utility_range", [0.2, 0.8])),
            "competitor_true_strength": self.rng.uniform(*scenario_cfg.get("competitor_strength_range", [0.1, 0.9])),
            "latent_risk_factor": self.rng.uniform(*scenario_cfg.get("latent_risk_factor_range", [0.0, 1.0])),
            "future_market_shock_step": self.rng.randint(1, max(1, horizon - 1)),
        }
        public_state = {
            "step": 0,
            "time_remaining": horizon,
            "public_market_signal": self.rng.choice(scenario_cfg.get("market_signals", ["stable", "volatile"])),
            "relationship_score": self.rng.uniform(*scenario_cfg.get("relationship_init_range", [0.3, 0.7])),
            "last_seller_message": "",
            "last_legal_warning": None,
            "last_risk_score": None,
        }

        initial_clause_templates = scenario_cfg.get(
            "initial_clause_templates",
            [
                {"clause_type": "pricing", "content": {"mode": "indexed"}},
                {"clause_type": "sla", "content": {"uptime_target": "configurable"}},
                {"clause_type": "liability", "content": {"cap_formula": "configurable"}},
            ],
        )

        initial_clauses = {}
        for c in initial_clause_templates:
            cid = self._id("cl")
            initial_clauses[cid] = Clause(
                clause_id=cid,
                clause_type=c["clause_type"],
                content=copy.deepcopy(c.get("content", {})),
                dependencies=copy.deepcopy(c.get("dependencies", [])),
            )

        return {
            "horizon": horizon,
            "hidden_state": hidden_state,
            "public_state": public_state,
            "initial_clauses": initial_clauses,
        }

    def _normalize_action(self, action: Action | Dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        return Action(action_type=ActionType(action["action_type"]), payload=action.get("payload", {}), metadata=action.get("metadata", {}))

    def _validate_action(self, action: Action) -> Tuple[bool, str]:
        schema = self.config.section("action_schema")
        required_fields = schema.get(action.action_type.value, {}).get("required_payload_fields", [])
        missing = [f for f in required_fields if f not in action.payload]
        if missing:
            return False, f"Missing payload fields: {missing}"
        return True, "ok"

    def _apply_buyer_action(self, action: Action) -> List[Dict[str, Any]]:
        notes = []
        if action.action_type == ActionType.PROPOSE_CONTRACT:
            parent = action.payload.get("contract_version_base", self.current_contract_id)
            clauses = self._clauses_from_payload(action.payload.get("clause_set", []))
            cv = self._new_contract_version(parent_id=parent, proposer="buyer", clauses=clauses)
            self.contract_versions[cv.version_id] = cv
            self.current_contract_id = cv.version_id
            notes.append({"type": "proposal_created", "version_id": cv.version_id})

        elif action.action_type == ActionType.MODIFY_CLAUSE:
            version = self.contract_versions[self.current_contract_id]
            target = action.payload.get("target_clause_id")
            op = action.payload.get("operation")
            if target in version.clauses and op == "replace":
                version.clauses[target].content = copy.deepcopy(action.payload.get("new_text_or_struct", {}))
                notes.append({"type": "clause_replaced", "clause_id": target})
            elif op == "add":
                cid = self._id("cl")
                version.clauses[cid] = Clause(
                    clause_id=cid,
                    clause_type=action.payload.get("clause_type", "custom"),
                    content=copy.deepcopy(action.payload.get("new_text_or_struct", {})),
                )
                notes.append({"type": "clause_added", "clause_id": cid})
            elif target in version.clauses and op == "remove":
                del version.clauses[target]
                notes.append({"type": "clause_removed", "clause_id": target})
            else:
                notes.append({"type": "clause_modify_noop", "reason": "invalid_target_or_op"})

        elif action.action_type == ActionType.DELAY_NEGOTIATION:
            delay_steps = int(action.payload.get("requested_duration", 1))
            self.events.append(
                Event(
                    event_id=self._id("evt"),
                    step_scheduled=self.step_count + delay_steps,
                    event_type="delayed_review_completed",
                    payload={"delay_steps": delay_steps},
                )
            )
            notes.append({"type": "delay_requested", "delay_steps": delay_steps})

        elif action.action_type == ActionType.ASK_QUESTION:
            notes.append({"type": "question_asked", "target": action.payload.get("target_agent", "seller")})

        elif action.action_type in (ActionType.ACCEPT_DEAL, ActionType.REJECT_DEAL):
            notes.append({"type": action.action_type.value})

        return notes

    def _update_public_state(
        self,
        action: Action,
        seller_out: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        fired_events: List[Dict[str, Any]],
    ) -> None:
        self.public_state["time_remaining"] = max(0, self.horizon - self.step_count - 1)
        self.public_state["last_seller_message"] = seller_out["message"]
        self.public_state["last_legal_warning"] = legal_out["severity"]
        self.public_state["last_risk_score"] = risk_out["risk_score"]

        rel_change_cfg = self.config.section("dynamics").get("relationship_deltas", {})
        action_key = action.action_type.value
        self.public_state["relationship_score"] = max(
            0.0,
            min(1.0, self.public_state["relationship_score"] + rel_change_cfg.get(action_key, 0.0)),
        )

        if fired_events:
            self.public_state["public_market_signal"] = "volatile"

    def _compute_reward(
        self,
        action_valid: bool,
        action: Action,
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        seller_out: Dict[str, Any],
        fired_events: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, float]]:
        w = self.config.section("reward_weights")

        progress = seller_out.get("concession_signal", 0.0)
        legal_penalty = 1.0 if legal_out.get("legal_warning") else 0.0
        risk_penalty = float(risk_out.get("risk_score", 0.0))
        time_penalty = 1.0 / max(1, self.public_state.get("time_remaining", 1))
        validity_penalty = 0.0 if action_valid else 1.0
        pressure_penalty = float(len(fired_events) > 0)

        terminal_bonus = 0.0
        if action.action_type == ActionType.ACCEPT_DEAL:
            terminal_bonus = 1.0 - (0.5 * legal_penalty + 0.5 * risk_penalty)
        elif action.action_type == ActionType.REJECT_DEAL:
            terminal_bonus = self.config.section("terminal").get("reject_bonus", 0.0)

        reward = (
            w.get("progress", 0.3) * progress
            - w.get("legal", 0.3) * legal_penalty
            - w.get("risk", 0.2) * risk_penalty
            - w.get("time", 0.1) * time_penalty
            - w.get("invalid_action", 0.1) * validity_penalty
            - w.get("pressure", 0.05) * pressure_penalty
            + w.get("terminal", 0.5) * terminal_bonus
        )

        breakdown = {
            "progress": progress,
            "legal_penalty": legal_penalty,
            "risk_penalty": risk_penalty,
            "time_penalty": time_penalty,
            "validity_penalty": validity_penalty,
            "pressure_penalty": pressure_penalty,
            "terminal_bonus": terminal_bonus,
        }
        return reward, breakdown

    def _is_terminated(self, action: Action, legal_out: Dict[str, Any], risk_out: Dict[str, Any]) -> bool:
        terminal_cfg = self.config.section("terminal")
        if action.action_type == ActionType.REJECT_DEAL and terminal_cfg.get("reject_ends_episode", True):
            return True

        if action.action_type == ActionType.ACCEPT_DEAL:
            max_legal = terminal_cfg.get("max_legal_severity_for_accept", ["none", "low", "medium"])
            max_risk = float(terminal_cfg.get("max_risk_for_accept", 0.8))
            return legal_out.get("severity", "high") in max_legal and float(risk_out.get("risk_score", 1.0)) <= max_risk

        catastrophic = float(risk_out.get("risk_score", 0.0)) >= float(terminal_cfg.get("catastrophic_risk_threshold", 0.98))
        return catastrophic

    def _build_observation(self) -> Dict[str, Any]:
        obs_cfg = self.config.section("observation")
        history_window = int(obs_cfg.get("history_window", 4))
        include_noise = bool(obs_cfg.get("enable_noise", True))

        risk_display = self.public_state.get("last_risk_score")
        if include_noise and risk_display is not None:
            risk_display = max(0.0, min(1.0, risk_display + self.rng.uniform(-0.05, 0.05)))

        visible = {
            "step": self.step_count,
            "time_remaining": self.public_state.get("time_remaining"),
            "market_signal": self.public_state.get("public_market_signal"),
            "relationship_score": self.public_state.get("relationship_score"),
            "current_contract": self._contract_summary(),
            "last_seller_message": self.public_state.get("last_seller_message"),
            "last_legal_warning": self.public_state.get("last_legal_warning"),
            "last_risk_score_noisy": risk_display,
            "available_actions": [a.value for a in ActionType],
            "recent_history": self.history[-history_window:],
        }
        return visible

    def _new_contract_version(self, parent_id: Optional[str], proposer: str, clauses: Dict[str, Clause]) -> ContractVersion:
        return ContractVersion(
            version_id=self._id("cv"),
            parent_version_id=parent_id,
            clauses=copy.deepcopy(clauses),
            proposer=proposer,
            step_created=self.step_count,
        )

    def _clauses_from_payload(self, clause_set: List[Dict[str, Any]]) -> Dict[str, Clause]:
        if not clause_set:
            return copy.deepcopy(self.contract_versions[self.current_contract_id].clauses)

        clauses: Dict[str, Clause] = {}
        for c in clause_set:
            cid = c.get("clause_id", self._id("cl"))
            clauses[cid] = Clause(
                clause_id=cid,
                clause_type=c.get("clause_type", "custom"),
                content=copy.deepcopy(c.get("content", {})),
                dependencies=copy.deepcopy(c.get("dependencies", [])),
            )
        return clauses

    def _fire_due_events(self) -> List[Dict[str, Any]]:
        due = [e for e in self.events if e.step_scheduled <= self.step_count]
        self.events = [e for e in self.events if e.step_scheduled > self.step_count]
        return [{"event_id": e.event_id, "event_type": e.event_type, **e.payload} for e in due]

    def _contract_summary(self) -> Dict[str, Any]:
        if not self.current_contract_id:
            return {"version_id": None, "clause_count": 0, "clause_types": []}
        cv = self.contract_versions[self.current_contract_id]
        return {
            "version_id": cv.version_id,
            "parent_version_id": cv.parent_version_id,
            "proposer": cv.proposer,
            "clause_count": len(cv.clauses),
            "clause_types": [c.clause_type for c in cv.clauses.values()],
        }

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
