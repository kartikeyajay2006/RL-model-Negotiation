from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RewardEngine:
    """Config-driven multi-component reward engine with delayed consequences.

    The engine consumes full environment state (including hidden state) to produce
    realistic rewards under partial observability for the learner.
    """

    cfg: Dict[str, Any]
    rng: random.Random

    @staticmethod
    def _action_name(action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_name", ""))
        return str(getattr(action, "action_name", ""))

    def initialize_reward_state(self) -> Dict[str, Any]:
        return {
            "pending_effects": [],
            "rolling_action_trace": [],
            "realized_long_term_penalty": 0.0,
            "running_discounted_reward": 0.0,
            "last_offer_price_signal": None,
            "last_trust_score": None,
            "last_reward": 0.0,
        }

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
        reward_state = state["reward_state"]

        self._update_pending_effects(
            action=action,
            state=state,
            legal_out=legal_out,
            risk_out=risk_out,
            step_index=step_index,
        )

        delayed_penalty_now = self._realize_pending_effects(reward_state, step_index)

        components = {
            "profit_optimization": self._profit_component(state, seller_out),
            "risk_penalty": self._risk_component(state, legal_out, risk_out, delayed_penalty_now),
            "time_efficiency": self._time_component(action, state, step_index, max_steps),
            "strategic_behavior": self._strategic_component(action, state, seller_out),
            "long_term_outcome": self._long_term_component(state, terminal_flags),
        }

        weighted = sum(float(weights.get(name, 0.0)) * float(value) for name, value in components.items())
        anti_exploit = self._anti_exploit_adjustment(action, state, components)
        reward = weighted + anti_exploit

        discount = float(reward_cfg.get("running_discount", 0.99))
        reward_state["running_discounted_reward"] = (
            discount * float(reward_state.get("running_discounted_reward", 0.0)) + reward
        )
        reward_state["last_reward"] = reward

        metadata = {
            "weighted_sum": weighted,
            "anti_exploit": anti_exploit,
            "delayed_penalty_realized": delayed_penalty_now,
            "pending_effects_count": len(reward_state.get("pending_effects", [])),
            "running_discounted_reward": reward_state["running_discounted_reward"],
        }
        return reward, components, metadata

    def _profit_component(self, state: Dict[str, Any], seller_out: Dict[str, Any]) -> float:
        cfg = self.cfg.get("reward", {}).get("profit_component", {})
        offer = state.get("current_offer", {})
        hidden = state.get("hidden_state", {})

        price_signal = float(offer.get("price_signal", 0.5))
        min_price = float(hidden.get("seller_private", {}).get("minimum_acceptable_price", 0.5))
        accept_prob = float(seller_out.get("accept_probability", 0.0))

        # Lower price is better for buyer, but only if likely to be feasible.
        gross_margin = 1.0 - price_signal
        feasibility = max(0.0, 1.0 - max(0.0, min_price - price_signal))
        expected_profit = gross_margin * (float(cfg.get("acceptance_importance", 0.6)) * accept_prob + (1.0 - float(cfg.get("acceptance_importance", 0.6))) * feasibility)

        return self._squash(expected_profit, cfg.get("squash_scale", 2.0))

    def _risk_component(
        self,
        state: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        delayed_penalty_now: float,
    ) -> float:
        cfg = self.cfg.get("reward", {}).get("risk_component", {})
        hidden = state.get("hidden_state", {})

        legal_risk = float(legal_out.get("legal_risk", 0.0))
        live_risk = float(risk_out.get("risk_score", 0.0))
        latent = float(hidden.get("latent_liability_factor", 0.0))

        combined = (
            float(cfg.get("live_risk_weight", 0.5)) * live_risk
            + float(cfg.get("legal_risk_weight", 0.35)) * legal_risk
            + float(cfg.get("latent_risk_weight", 0.15)) * latent
            + float(cfg.get("delayed_effect_weight", 1.0)) * delayed_penalty_now
        )

        return -self._squash(combined, cfg.get("squash_scale", 2.5))

    def _time_component(self, action: Any, state: Dict[str, Any], step_index: int, max_steps: int) -> float:
        cfg = self.cfg.get("reward", {}).get("time_component", {})
        frac_used = float(step_index + 1) / max(1.0, float(max_steps))
        action_name = self._action_name(action)

        urgency_penalty = frac_used ** float(cfg.get("urgency_exponent", 1.5))
        delay_penalty = float(cfg.get("delay_action_penalty", 0.15)) if action_name == "delay_negotiation" else 0.0
        question_bonus = float(cfg.get("timely_question_bonus", 0.05)) if action_name == "ask_question" and frac_used < float(cfg.get("question_bonus_cutoff", 0.4)) else 0.0

        return question_bonus - urgency_penalty - delay_penalty

    def _strategic_component(self, action: Any, state: Dict[str, Any], seller_out: Dict[str, Any]) -> float:
        cfg = self.cfg.get("reward", {}).get("strategic_component", {})
        rs = state["reward_state"]
        action_name = self._action_name(action)

        trace: List[str] = rs.setdefault("rolling_action_trace", [])
        trace.append(action_name)
        max_trace = int(cfg.get("trace_window", 6))
        if len(trace) > max_trace:
            del trace[:-max_trace]

        diversity = len(set(trace)) / max(1.0, float(len(trace)))
        seller_alignment = float(seller_out.get("accept_probability", 0.0))

        repetitive_penalty = 0.0
        if len(trace) >= int(cfg.get("repetition_check_window", 3)):
            tail = trace[-int(cfg.get("repetition_check_window", 3)):]
            if len(set(tail)) == 1:
                repetitive_penalty = float(cfg.get("repetition_penalty", 0.1))

        return (
            float(cfg.get("diversity_weight", 0.5)) * diversity
            + float(cfg.get("alignment_weight", 0.5)) * seller_alignment
            - repetitive_penalty
        )

    def _long_term_component(self, state: Dict[str, Any], terminal_flags: Dict[str, bool]) -> float:
        cfg = self.cfg.get("reward", {}).get("long_term_component", {})
        trust = float(state.get("trust_score", 0.0))
        pressure = float(state.get("external_pressure", 0.0))
        hidden = state.get("hidden_state", {})
        rs = state["reward_state"]

        outstanding = sum(float(e.get("magnitude", 0.0)) for e in rs.get("pending_effects", []))
        carry_penalty = float(cfg.get("outstanding_penalty_weight", 0.2)) * outstanding

        quality = float(cfg.get("trust_weight", 0.7)) * trust - float(cfg.get("pressure_weight", 0.3)) * pressure - carry_penalty

        terminal_adjust = 0.0
        if terminal_flags.get("deal_success", False):
            terminal_adjust += float(cfg.get("success_scale", 1.0)) * (trust - float(hidden.get("latent_liability_factor", 0.0)))
        if terminal_flags.get("failure", False):
            terminal_adjust -= float(cfg.get("failure_scale", 1.0)) * (1.0 - trust + outstanding)
        if terminal_flags.get("deadline", False):
            terminal_adjust -= float(cfg.get("deadline_scale", 0.5)) * (1.0 + pressure)

        return quality + terminal_adjust

    def _anti_exploit_adjustment(self, action: Any, state: Dict[str, Any], components: Dict[str, float]) -> float:
        cfg = self.cfg.get("reward", {}).get("anti_exploit", {})
        rs = state["reward_state"]

        prev = float(rs.get("last_reward", 0.0))
        mean_abs_component = sum(abs(v) for v in components.values()) / max(1.0, float(len(components)))
        oscillation = abs(components["strategic_behavior"] - prev)

        clipping = float(cfg.get("component_clip", 3.0))
        stability_penalty = max(0.0, oscillation - float(cfg.get("oscillation_tolerance", 0.8)))

        bounded = max(-clipping, min(clipping, mean_abs_component))
        return -float(cfg.get("oscillation_weight", 0.05)) * stability_penalty + float(cfg.get("bounded_bonus_weight", 0.02)) * (1.0 - abs(bounded) / max(1.0, clipping))

    def _update_pending_effects(
        self,
        *,
        action: Any,
        state: Dict[str, Any],
        legal_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        step_index: int,
    ) -> None:
        cfg = self.cfg.get("reward", {}).get("delayed_effects", {})
        rs = state["reward_state"]
        hidden = state.get("hidden_state", {})
        action_name = self._action_name(action)

        trigger_strength = (
            float(cfg.get("risk_trigger_weight", 0.5)) * float(risk_out.get("risk_score", 0.0))
            + float(cfg.get("legal_trigger_weight", 0.3)) * float(legal_out.get("legal_risk", 0.0))
            + float(cfg.get("latent_trigger_weight", 0.2)) * float(hidden.get("latent_liability_factor", 0.0))
        )
        if action_name == "accept_deal":
            trigger_strength *= float(cfg.get("accept_multiplier", 1.4))

        spawn_prob = max(0.0, min(1.0, trigger_strength * float(cfg.get("spawn_scale", 0.8))))
        if self.rng.random() > spawn_prob:
            return

        delay_min, delay_max = cfg.get("delay_range", [1, 4])
        delay_steps = self.rng.randint(int(delay_min), int(delay_max))
        mag_min, mag_max = cfg.get("magnitude_range", [0.01, 0.20])
        magnitude = self.rng.uniform(float(mag_min), float(mag_max)) * (1.0 + trigger_strength)

        rs.setdefault("pending_effects", []).append(
            {
                "trigger_step": int(step_index),
                "apply_step": int(step_index + delay_steps),
                "magnitude": float(magnitude),
                "kind": "hidden_contract_liability",
            }
        )

    def _realize_pending_effects(self, reward_state: Dict[str, Any], step_index: int) -> float:
        pending = reward_state.setdefault("pending_effects", [])
        matured = [e for e in pending if int(e.get("apply_step", 10**9)) <= int(step_index)]
        pending[:] = [e for e in pending if int(e.get("apply_step", 10**9)) > int(step_index)]

        penalty = sum(float(e.get("magnitude", 0.0)) for e in matured)
        reward_state["realized_long_term_penalty"] = float(reward_state.get("realized_long_term_penalty", 0.0)) + penalty
        return penalty

    @staticmethod
    def _squash(x: float, scale: float) -> float:
        if scale <= 0:
            return x
        x = max(-50.0, min(50.0, x * scale))
        return math.tanh(x)
