from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List


def _safe_sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SellerAgent:
    cfg: Dict[str, Any]
    rng: random.Random

    def initialize_hidden_state(self) -> Dict[str, Any]:
        return {
            "minimum_acceptable_price": self.rng.uniform(*self.cfg.get("minimum_price_range", [0.2, 0.9])),
            "walkaway_tolerance": self.rng.uniform(*self.cfg.get("walkaway_tolerance_range", [0.1, 0.7])),
            "bluff_factor": self.rng.uniform(*self.cfg.get("bluff_factor_range", [0.0, 1.0])),
            "strategy_temperature": self.rng.uniform(*self.cfg.get("temperature_range", [0.5, 2.0])),
        }

    def respond(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        hidden = shared_state["hidden_state"]
        offer = shared_state.get("current_offer") or {}

        proposed_price = float(offer.get("price_signal", self.rng.random()))
        min_price = float(hidden["seller_private"]["minimum_acceptable_price"])
        pressure = float(shared_state["external_pressure"])
        trust = float(shared_state["trust_score"])
        temp = float(hidden["seller_private"]["strategy_temperature"])

        # Acceptance probability is smooth and stochastic, driven by latent utility.
        utility_gap = (proposed_price - min_price) + 0.25 * trust - 0.15 * pressure
        accept_prob = _safe_sigmoid(utility_gap / max(0.1, temp))

        counter_direction = 1.0 if proposed_price < min_price else -1.0
        counter_delta = counter_direction * self.rng.uniform(*self.cfg.get("counter_delta_range", [0.01, 0.25]))
        counter_price = max(0.0, min(1.0, proposed_price + counter_delta))

        mode_sample = self.rng.random()
        reject_floor = float(self.cfg.get("reject_floor_probability", 0.05))

        if mode_sample < accept_prob:
            decision = "accept"
        elif mode_sample < min(1.0, accept_prob + reject_floor):
            decision = "reject"
        else:
            decision = "counter"

        return {
            "decision": decision,
            "accept_probability": accept_prob,
            "counter_offer": {"price_signal": counter_price, "clause_bias": self.rng.uniform(-1.0, 1.0)},
            "message_style": self.rng.choice(self.cfg.get("message_styles", ["direct", "guarded", "diplomatic"])),
        }


@dataclass
class LegalAgent:
    cfg: Dict[str, Any]
    rng: random.Random

    def evaluate(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        offer = shared_state.get("current_offer") or {}
        clauses = offer.get("clauses", [])

        complexity = float(len(clauses))
        ambiguity = sum(float(c.get("ambiguity", self.rng.random())) for c in clauses) / max(1.0, complexity)
        missing_fields = sum(1 for c in clauses if not c.get("required_terms_complete", False))

        risk_noise = self.rng.gauss(0.0, float(self.cfg.get("risk_noise_std", 0.08)))
        base = float(self.cfg.get("base_legal_risk", 0.2))
        legal_risk = max(
            0.0,
            min(1.0, base + 0.15 * ambiguity + 0.07 * missing_fields + 0.03 * complexity + risk_noise),
        )

        penalty_scale = float(self.cfg.get("penalty_scale", 1.0))
        penalties = {
            "legal_risk": legal_risk,
            "penalty": penalty_scale * legal_risk,
            "required_revisions": [
                {"clause_id": c.get("id", "unknown"), "issue": "incomplete_or_ambiguous"}
                for c in clauses
                if (not c.get("required_terms_complete", False)) or float(c.get("ambiguity", 0.0)) > self.rng.uniform(0.3, 0.9)
            ],
        }
        return penalties


@dataclass
class RiskAgent:
    cfg: Dict[str, Any]
    rng: random.Random

    def evaluate(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        history = shared_state.get("negotiation_history", [])
        hidden = shared_state["hidden_state"]

        action_volatility = sum(abs(float(h.get("volatility_signal", 0.0))) for h in history[-5:]) / max(1, len(history[-5:]))
        uncertainty = float(hidden.get("market_uncertainty", 0.5))
        pressure = float(shared_state.get("external_pressure", 0.0))

        inertia = float(self.cfg.get("inertia", 0.7))
        prev = float(shared_state.get("risk_score", 0.0))
        innovation = (
            float(self.cfg.get("base_risk", 0.2))
            + float(self.cfg.get("uncertainty_weight", 0.4)) * uncertainty
            + float(self.cfg.get("volatility_weight", 0.25)) * action_volatility
            + float(self.cfg.get("pressure_weight", 0.15)) * pressure
            + self.rng.gauss(0.0, float(self.cfg.get("noise_std", 0.05)))
        )
        risk_score = max(0.0, min(1.0, inertia * prev + (1.0 - inertia) * innovation))

        return {
            "risk_score": risk_score,
            "uncertainty_component": uncertainty,
            "volatility_component": action_volatility,
        }


@dataclass
class CompetitorAgent:
    cfg: Dict[str, Any]
    rng: random.Random

    def pressure_update(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        hidden = shared_state["hidden_state"]
        p_event = float(self.cfg.get("event_probability", 0.2))
        shock_scale = float(self.cfg.get("shock_scale", 0.2))

        event_triggered = self.rng.random() < p_event
        latent_competition = float(hidden.get("competitor_intensity", 0.5))
        shock = self.rng.uniform(-shock_scale, shock_scale) if event_triggered else self.rng.uniform(-0.02, 0.02)

        pressure_delta = 0.3 * latent_competition + shock
        new_pressure = max(0.0, min(1.0, float(shared_state.get("external_pressure", 0.0)) * 0.6 + pressure_delta * 0.4))

        return {
            "external_pressure": new_pressure,
            "event_triggered": event_triggered,
            "event_intensity": max(0.0, min(1.0, abs(shock))),
        }
