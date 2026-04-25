from __future__ import annotations

import copy
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .internal_agents import CompetitorAgent, LegalAgent, RiskAgent, SellerAgent
from .reward_engine import RewardEngine


class DealForgeMCPEnvironment:
    """Standalone DealForge environment (no OpenEnv dependency)."""

    def __init__(self, config: Dict[str, Any] | str, seed: Optional[int] = None) -> None:
        self.config = self._load_config(config)
        self.rng = random.Random(seed)

        self._episode_id: Optional[str] = None
        self._step_count = 0
        self._max_steps = 0
        self._done = False
        self._state_dict: Dict[str, Any] = {}

        self.seller_agent = SellerAgent(self.config.get("seller_agent", {}), self.rng)
        self.legal_agent = LegalAgent(self.config.get("legal_agent", {}), self.rng)
        self.risk_agent = RiskAgent(self.config.get("risk_agent", {}), self.rng)
        self.competitor_agent = CompetitorAgent(self.config.get("competitor_agent", {}), self.rng)
        self.reward_engine = RewardEngine(self.config, self.rng)

    @staticmethod
    def _load_config(config: Dict[str, Any] | str) -> Dict[str, Any]:
        if isinstance(config, dict):
            return copy.deepcopy(config)
        cfg_path = Path(config)
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Initialize a new negotiation episode and return initial visible state."""
        if seed is not None:
            self.rng.seed(seed)

        scenario_cfg = self.config.get("scenario", {})
        horizon_min, horizon_max = scenario_cfg.get("horizon_range", [10, 30])
        self._max_steps = self.rng.randint(int(horizon_min), int(horizon_max))

        self._episode_id = episode_id or f"dealforge-{uuid.uuid4().hex[:10]}"
        self._step_count = 0
        self._done = False

        hidden_state = {
            "seller_private": self.seller_agent.initialize_hidden_state(),
            "market_uncertainty": self.rng.uniform(*scenario_cfg.get("market_uncertainty_range", [0.1, 0.9])),
            "competitor_intensity": self.rng.uniform(*scenario_cfg.get("competitor_intensity_range", [0.1, 0.9])),
            "relationship_sensitivity": self.rng.uniform(*scenario_cfg.get("relationship_sensitivity_range", [0.2, 1.2])),
            "latent_liability_factor": self.rng.uniform(*scenario_cfg.get("latent_liability_factor_range", [0.0, 1.0])),
        }

        initial_offer = {
            "price_signal": self.rng.uniform(*scenario_cfg.get("initial_price_signal_range", [0.2, 0.8])),
            "clauses": copy.deepcopy(scenario_cfg.get("initial_clauses", [])),
            "version": 0,
        }

        self._state_dict = {
            "negotiation_history": [],
            "current_offer": initial_offer,
            "time_steps": {"current": 0, "remaining": self._max_steps, "max": self._max_steps},
            "risk_score": self.rng.uniform(*scenario_cfg.get("initial_risk_score_range", [0.1, 0.6])),
            "trust_score": self.rng.uniform(*scenario_cfg.get("initial_trust_score_range", [0.2, 0.8])),
            "external_pressure": self.rng.uniform(*scenario_cfg.get("initial_external_pressure_range", [0.0, 0.4])),
            "visible_state": {},
            "hidden_state": hidden_state,
            "reward_state": self.reward_engine.initialize_reward_state(),
            "terminal": {"deal_success": False, "deadline": False, "failure": False},
        }

        self._refresh_visible_state(last_event={"event": "reset"})
        return copy.deepcopy(self._state_dict["visible_state"])

    def step(self, action: Dict[str, Any] | Any) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one environment step and return (observation, reward, done, info)."""
        if self._done:
            info = {"error": "Episode has ended. Call reset().", "episode_id": self._episode_id}
            return copy.deepcopy(self._state_dict.get("visible_state", {})), 0.0, True, info

        norm_action = self._normalize_action(action)
        transition_event = self._apply_agent_action(norm_action)

        seller_out = self.seller_agent.respond(self._state_dict)
        legal_out = self.legal_agent.evaluate(self._state_dict)
        risk_out = self.risk_agent.evaluate(self._state_dict)
        competitor_out = self.competitor_agent.pressure_update(self._state_dict)

        self._merge_internal_outputs(seller_out, legal_out, risk_out, competitor_out)

        self._step_count += 1
        self._state_dict["time_steps"] = {
            "current": self._step_count,
            "remaining": max(0, self._max_steps - self._step_count),
            "max": self._max_steps,
        }

        terminal_reasons = self._compute_termination(norm_action, seller_out, legal_out, risk_out)
        self._done = any(terminal_reasons.values())
        self._state_dict["terminal"] = terminal_reasons

        reward, reward_components, reward_meta = self.reward_engine.compute_step_reward(
            action=norm_action,
            state=self._state_dict,
            seller_out=seller_out,
            legal_out=legal_out,
            risk_out=risk_out,
            competitor_out=competitor_out,
            terminal_flags=terminal_reasons,
            step_index=self._step_count,
            max_steps=self._max_steps,
        )

        step_event = {
            "event": "step",
            "action": norm_action["action_name"],
            "action_payload": copy.deepcopy(norm_action["payload"]),
            "seller": seller_out,
            "legal": legal_out,
            "risk": risk_out,
            "competitor": competitor_out,
            "volatility_signal": self.rng.uniform(-1.0, 1.0),
        }
        self._state_dict["negotiation_history"].append(step_event)

        self._refresh_visible_state(last_event=transition_event)

        info = {
            "episode_id": self._episode_id,
            "termination": terminal_reasons,
            "reward_components": reward_components,
            "reward_metadata": reward_meta,
            "event_log": [transition_event, step_event],
        }
        return copy.deepcopy(self._state_dict["visible_state"]), float(reward), self._done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state, including hidden variables."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "dynamic_state": copy.deepcopy(self._state_dict),
        }

    def close(self) -> None:
        """Release resources and clear state."""
        self._state_dict = {}
        self._episode_id = None
        self._done = True

    def _normalize_action(self, action: Dict[str, Any] | Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            if "action_name" in action:
                return {"action_name": action.get("action_name"), "payload": action.get("payload", {})}
            if "action_type" in action:
                return {"action_name": action.get("action_type"), "payload": action.get("payload", {})}
        action_name = getattr(action, "action_name", None)
        payload = getattr(action, "payload", {})
        return {"action_name": action_name, "payload": payload}

    def _apply_agent_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        current_offer = self._state_dict.get("current_offer", {})
        next_offer = copy.deepcopy(current_offer)
        next_offer["version"] = int(current_offer.get("version", 0)) + 1

        action_name = action.get("action_name")
        payload = action.get("payload", {})

        if action_name == "propose_contract":
            next_offer.update(copy.deepcopy(payload))
        elif action_name == "modify_clause":
            clauses = next_offer.get("clauses", [])
            clauses.append(copy.deepcopy(payload.get("clause_patch", {})))
            next_offer["clauses"] = clauses
        elif action_name == "delay_negotiation":
            drift = self.config.get("dynamics", {}).get("delay_pressure_drift", [0.0, 0.08])
            self._state_dict["external_pressure"] = max(
                0.0,
                min(1.0, float(self._state_dict["external_pressure"]) + self.rng.uniform(float(drift[0]), float(drift[1]))),
            )
        elif action_name == "ask_question":
            info_gain = self.rng.uniform(*self.config.get("dynamics", {}).get("question_information_gain", [0.01, 0.10]))
            self._state_dict["hidden_state"]["market_uncertainty"] = max(
                0.0,
                float(self._state_dict["hidden_state"]["market_uncertainty"]) - info_gain,
            )

        self._state_dict["current_offer"] = next_offer
        return {
            "event": "agent_action_applied",
            "action_name": action_name,
            "offer_version": next_offer.get("version", 0),
        }

    def _merge_internal_outputs(
        self,
        seller_out: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        competitor_out: Dict[str, Any],
    ) -> None:
        if seller_out.get("decision") == "counter":
            self._state_dict["current_offer"].update(copy.deepcopy(seller_out.get("counter_offer", {})))

        trust_cfg = self.config.get("dynamics", {}).get("trust_update", {})
        trust = float(self._state_dict["trust_score"])
        trust += float(trust_cfg.get(seller_out.get("decision", "counter"), 0.0))
        trust -= float(legal_out.get("penalty", 0.0)) * float(trust_cfg.get("legal_penalty_weight", 0.05))
        trust = max(0.0, min(1.0, trust))

        self._state_dict["trust_score"] = trust
        self._state_dict["risk_score"] = float(risk_out.get("risk_score", self._state_dict["risk_score"]))
        self._state_dict["external_pressure"] = float(competitor_out.get("external_pressure", self._state_dict["external_pressure"]))

    def _compute_termination(
        self,
        action: Dict[str, Any],
        seller_out: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
    ) -> Dict[str, bool]:
        term_cfg = self.config.get("termination", {})

        deal_success = False
        if action.get("action_name") == "accept_deal":
            accept_prob = float(seller_out.get("accept_probability", 0.0))
            accept_sensitivity = float(term_cfg.get("accept_sensitivity", 1.0))
            legal_risk = float(legal_out.get("legal_risk", 1.0))
            total_risk = float(risk_out.get("risk_score", 1.0))

            success_likelihood = max(0.0, min(1.0, accept_prob * accept_sensitivity * (1.0 - 0.5 * (legal_risk + total_risk))))
            deal_success = self.rng.random() < success_likelihood

        deadline = self._step_count >= self._max_steps

        failure_risk = (
            float(term_cfg.get("risk_weight", 0.5)) * float(risk_out.get("risk_score", 0.0))
            + float(term_cfg.get("pressure_weight", 0.3)) * float(self._state_dict.get("external_pressure", 0.0))
            + float(term_cfg.get("trust_weight", 0.2)) * (1.0 - float(self._state_dict.get("trust_score", 0.0)))
        )
        failure = self.rng.random() < max(0.0, min(1.0, failure_risk))

        if action.get("action_name") == "reject_deal":
            failure = self.rng.random() < float(term_cfg.get("reject_failure_probability", 0.7))

        return {
            "deal_success": bool(deal_success),
            "deadline": bool(deadline and not deal_success),
            "failure": bool(failure and not deal_success),
        }

    def _refresh_visible_state(self, last_event: Dict[str, Any]) -> None:
        history = self._state_dict.get("negotiation_history", [])
        visible = {
            "negotiation_history": copy.deepcopy(history[-int(self.config.get("observation", {}).get("history_window", 6)):]),
            "current_offer": copy.deepcopy(self._state_dict.get("current_offer", {})),
            "time_steps": copy.deepcopy(self._state_dict.get("time_steps", {})),
            "risk_score": float(self._state_dict.get("risk_score", 0.0)),
            "trust_score": float(self._state_dict.get("trust_score", 0.0)),
            "external_pressure": float(self._state_dict.get("external_pressure", 0.0)),
            "last_event": copy.deepcopy(last_event),
            "available_actions": [
                "propose_contract",
                "modify_clause",
                "accept_deal",
                "reject_deal",
                "delay_negotiation",
                "ask_question",
            ],
        }
        self._state_dict["visible_state"] = visible
