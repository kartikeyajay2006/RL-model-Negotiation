from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RewardEngine:
    """Dense, multi-component reward engine for negotiation learning."""

    cfg: Dict[str, Any]
    rng: random.Random

    def initialize_reward_state(self) -> Dict[str, Any]:
        return {
            "last_reward": 0.0,
            "running_discounted_reward": 0.0,
            "baseline_price": None,
            "previous_price_signal": None,
            "previous_price_gap": None,
            "last_action_name": None,
            "offer_delta_trace": [],
            "pending_effects": [],
            "realized_long_term_penalty": 0.0,
            "trust_score": 0.5,
            "delay_action_count": 0,
            "action_trace": [],
        }

    @staticmethod
    def _action_name(action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_name", ""))
        return str(getattr(action, "action_name", ""))

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _price_signal(self, state: Dict[str, Any]) -> float:
        offer = state.get("current_offer", {})
        return self._clip(self._safe_float(offer.get("price_signal", 0.5), 0.5), 1e-6, 1e6)

    def _seller_anchor(self, state: Dict[str, Any]) -> float:
        hidden = state.get("hidden_state", {})
        seller_private = hidden.get("seller_private", {})
        return self._clip(self._safe_float(seller_private.get("minimum_acceptable_price", 0.5), 0.0), 0.0, 1e6)

    def _is_invalid_action(self, action: Any, state: Dict[str, Any]) -> bool:
        reward_cfg = self.cfg.get("reward", {})
        invalid_cfg = reward_cfg.get("invalid_action", {})
        action_name = self._action_name(action)
        payload = action.get("payload", {}) if isinstance(action, dict) else getattr(action, "payload", {})

        valid_actions = invalid_cfg.get(
            "valid_actions",
            [
                "propose_contract",
                "modify_clause",
                "accept_deal",
                "reject_deal",
                "delay_negotiation",
                "ask_question",
            ],
        )
        if action_name not in valid_actions:
            return True

        if not isinstance(payload, dict):
            return True

        require_clause_patch = bool(invalid_cfg.get("require_clause_patch_for_modify", True))
        if action_name == "modify_clause" and require_clause_patch and "clause_patch" not in payload:
            return True

        require_target_version = bool(invalid_cfg.get("require_target_version_for_terminal_actions", False))
        if action_name in {"accept_deal", "reject_deal"} and require_target_version and "target_version" not in payload:
            return True

        require_price_signal = bool(invalid_cfg.get("require_price_signal_for_propose", False))
        if action_name == "propose_contract" and require_price_signal and "price_signal" not in payload:
            return True

        _ = state
        return False

    def _update_delayed_effects(
        self,
        *,
        action_name: str,
        state: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        step_index: int,
    ) -> None:
        delayed_cfg = self.cfg.get("reward", {}).get("delayed_effects", {})
        rs = state["reward_state"]
        hidden = state.get("hidden_state", {})

        trigger = (
            self._safe_float(delayed_cfg.get("risk_trigger_weight", 0.5), 0.5) * self._safe_float(risk_out.get("risk_score", 0.0), 0.0)
            + self._safe_float(delayed_cfg.get("legal_trigger_weight", 0.3), 0.3) * self._safe_float(legal_out.get("legal_risk", 0.0), 0.0)
            + self._safe_float(delayed_cfg.get("latent_trigger_weight", 0.2), 0.2) * self._safe_float(hidden.get("latent_liability_factor", 0.0), 0.0)
        )

        if action_name == "accept_deal":
            trigger *= self._safe_float(delayed_cfg.get("accept_multiplier", 1.4), 1.4)

        spawn_scale = self._safe_float(delayed_cfg.get("spawn_scale", 0.8), 0.8)
        spawn_prob = self._clip(trigger * spawn_scale, 0.0, 1.0)
        if self.rng.random() > spawn_prob:
            return

        delay_range = delayed_cfg.get("delay_range", [1, 4])
        mag_range = delayed_cfg.get("magnitude_range", [0.01, 0.2])
        delay_steps = self.rng.randint(int(delay_range[0]), int(delay_range[1]))
        magnitude = self.rng.uniform(float(mag_range[0]), float(mag_range[1])) * (1.0 + trigger)

        rs.setdefault("pending_effects", []).append(
            {
                "trigger_step": int(step_index),
                "apply_step": int(step_index + delay_steps),
                "magnitude": float(magnitude),
                "kind": "latent_contract_liability",
            }
        )

    def _realize_delayed_penalty(self, reward_state: Dict[str, Any], step_index: int) -> float:
        pending = reward_state.setdefault("pending_effects", [])
        matured = [e for e in pending if int(e.get("apply_step", 10**9)) <= int(step_index)]
        pending[:] = [e for e in pending if int(e.get("apply_step", 10**9)) > int(step_index)]
        penalty = sum(self._safe_float(e.get("magnitude", 0.0), 0.0) for e in matured)
        reward_state["realized_long_term_penalty"] = self._safe_float(
            reward_state.get("realized_long_term_penalty", 0.0), 0.0
        ) + penalty
        return float(penalty)

    def compute_step_reward(
        self,
        *,
        action: Any,
        state: Dict[str, Any],
        seller_out: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        competitor_out: Dict[str, Any],
        terminal_flags: Dict[str, bool],
        step_index: int,
        max_steps: int,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        reward_cfg = self.cfg.get("reward", {})
        weights = reward_cfg.get("weights", {})
        rs = state["reward_state"]

        action_name = self._action_name(action)
        price = self._price_signal(state)
        seller_anchor = self._seller_anchor(state)
        previous_price = self._safe_float(rs.get("previous_price_signal", price), price)
        previous_gap = self._safe_float(
            rs.get("previous_price_gap", abs(previous_price - seller_anchor)),
            abs(previous_price - seller_anchor),
        )
        current_gap = abs(price - seller_anchor)
        progress = max(previous_gap - current_gap, 0.0)

        counter_offer = seller_out.get("counter_offer", {}) if isinstance(seller_out, dict) else {}
        seller_counter = self._safe_float(counter_offer.get("price_signal", seller_anchor), seller_anchor)
        fairness = self._clip(1.0 - abs(price - seller_counter), 0.0, 1.0)

        deal_closure = 1.0 if bool(terminal_flags.get("deal_success", False)) else 0.0
        risk_score = self._clip(
            self._safe_float(risk_out.get("risk_score", state.get("risk_score", 0.0)), 0.0),
            0.0,
            1.0,
        )

        rs["delay_action_count"] = int(rs.get("delay_action_count", 0))
        if action_name == "delay_negotiation":
            rs["delay_action_count"] += 1
        delay_scale = self._safe_float(reward_cfg.get("delay_penalty_scale", 0.2), 0.2)
        delay_penalty = delay_scale * float(rs["delay_action_count"])

        trace: List[str] = rs.setdefault("action_trace", [])
        trace.append(action_name)
        if len(trace) > 6:
            del trace[:-6]
        repeat_penalty = 0.0
        if len(trace) >= 3 and len(set(trace[-3:])) == 1:
            repeat_penalty = self._safe_float(reward_cfg.get("repeat_penalty_value", 0.3), 0.3)

        invalid_scale = self._safe_float(reward_cfg.get("invalid_action_penalty", 0.5), 0.5)
        invalid_action_penalty = invalid_scale if self._is_invalid_action(action, state) else 0.0

        value_reward = self._clip((seller_anchor - price) / max(1e-6, seller_anchor), -1.0, 1.0)
        agreement_reward = progress * self._safe_float(reward_cfg.get("agreement_progress_scale", 0.8), 0.8)
        trust_reward = fairness
        risk_penalty = (risk_score ** 2) * self._safe_float(reward_cfg.get("risk_penalty_scale", 2.0), 2.0)

        total_before_norm = (
            self._safe_float(weights.get("deal_closure", 2.0), 2.0) * deal_closure
            + self._safe_float(weights.get("fairness", 1.2), 1.2) * fairness
            + self._safe_float(weights.get("progress", 1.5), 1.5) * progress
            - self._safe_float(weights.get("risk_penalty", 1.0), 1.0) * (risk_score ** 2)
            - delay_penalty
            - repeat_penalty
        )
        deal_rejected = bool(action_name == "reject_deal" or terminal_flags.get("failure", False) or terminal_flags.get("deadline", False))
        if bool(terminal_flags.get("deal_success", False)):
            total_before_norm += self._safe_float(reward_cfg.get("success_bonus", 1.0), 1.0)
        elif deal_rejected:
            total_before_norm -= abs(self._safe_float(reward_cfg.get("failure_penalty", 0.5), 0.5))
        reward = self._clip(total_before_norm, -2.0, 2.0)

        components = {
            "deal_closure": float(deal_closure),
            "fairness": float(fairness),
            "progress": float(progress),
            "value_reward": float(value_reward),
            "agreement_reward": float(agreement_reward),
            "trust_reward": float(trust_reward),
            "risk_penalty": float(risk_penalty),
            "delay_penalty": float(delay_penalty),
            "inefficiency_penalty": float(delay_penalty),
            "repeat_penalty": float(repeat_penalty),
            "invalid_action_penalty": float(invalid_action_penalty),
            "invalid_penalty": float(invalid_action_penalty),
            "success_reward": float(deal_closure),
            "deal_bonus": float(0.0),
            "pressure_penalty": float(0.0),
            "oscillation_penalty": float(0.0),
        }

        running_discount = self._safe_float(reward_cfg.get("running_discount", 0.99), 0.99)
        rs["running_discounted_reward"] = (
            running_discount * self._safe_float(rs.get("running_discounted_reward", 0.0), 0.0) + reward
        )
        rs["last_reward"] = reward
        rs["previous_price_signal"] = price
        rs["previous_price_gap"] = current_gap
        rs["last_action_name"] = action_name

        metadata = {
            "total_before_clip": float(total_before_norm),
            "clip_min": float(-2.0),
            "clip_max": float(2.0),
            "step_index": int(step_index),
            "max_steps": int(max_steps),
            "price_signal": float(price),
            "baseline_price": float(self._safe_float(rs.get("baseline_price", price), price)),
            "price_gap": float(current_gap),
            "progress_delta": float(progress),
            "delayed_penalty_realized": 0.0,
            "pending_effects_count": 0,
            "running_discounted_reward": float(rs["running_discounted_reward"]),
            "terminal_flags": dict(terminal_flags),
            "seller_accept_probability": self._safe_float(seller_out.get("accept_probability", 0.0), 0.0),
            "legal_risk": self._safe_float(legal_out.get("legal_risk", 0.0), 0.0),
            "external_pressure": self._safe_float(
                competitor_out.get("external_pressure", state.get("external_pressure", 0.0)),
                0.0,
            ),
            "risk_score": float(risk_score),
            "trust_score": float(fairness),
        }

        return float(reward), components, metadata
