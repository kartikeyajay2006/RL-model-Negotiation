from __future__ import annotations

import random
from typing import Any, Dict, List


class BaseInternalAgent:
    def __init__(self, name: str, cfg: Dict[str, Any], rng: random.Random):
        self.name = name
        self.cfg = cfg
        self.rng = rng


class SellerAgent(BaseInternalAgent):
    def respond(self, public_state: Dict[str, Any], hidden_state: Dict[str, Any]) -> Dict[str, Any]:
        mood = self.rng.choices(
            self.cfg.get("moods", ["cooperative", "aggressive", "neutral"]),
            weights=self.cfg.get("mood_weights", [0.4, 0.3, 0.3]),
            k=1,
        )[0]
        concession = self.rng.random() * self.cfg.get("max_concession_scale", 0.2)
        return {"mood": mood, "concession_signal": concession, "message": "seller_response"}


class LegalAgent(BaseInternalAgent):
    def evaluate(self, contract_summary: Dict[str, Any]) -> Dict[str, Any]:
        violation_prob = self.cfg.get("base_violation_prob", 0.2)
        has_violation = self.rng.random() < violation_prob
        severity = self.rng.choice(self.cfg.get("severity_levels", ["low", "medium", "high"]))
        return {
            "legal_warning": has_violation,
            "severity": severity if has_violation else "none",
            "required_changes": [] if not has_violation else ["revise_indemnity", "clarify_sla"],
        }


class RiskAgent(BaseInternalAgent):
    def evaluate(self, public_state: Dict[str, Any], hidden_state: Dict[str, Any]) -> Dict[str, Any]:
        base = self.cfg.get("base_risk", 0.3)
        drift = self.cfg.get("risk_drift_per_step", 0.01) * public_state["step"]
        noise = self.rng.uniform(-0.1, 0.1)
        score = max(0.0, min(1.0, base + drift + noise))
        return {"risk_score": score, "risk_flags": ["delivery", "compliance"] if score > 0.6 else ["delivery"]}


class CompetitorAgent(BaseInternalAgent):
    def maybe_inject(self, public_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        p = self.cfg.get("event_probability", 0.2)
        if self.rng.random() > p:
            return []
        intensity = self.rng.choice(self.cfg.get("intensity_levels", ["mild", "moderate", "severe"]))
        return [{"event_type": "competitor_pressure", "intensity": intensity}]
