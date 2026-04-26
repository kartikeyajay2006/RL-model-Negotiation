from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class RewardEngine:
    """Composable reward engine for deterministic negotiation episodes."""

    cfg: Dict[str, Any]
    rng: Any

    def initialize_reward_state(self) -> Dict[str, Any]:
        return {
            "last_reward": 0.0,
            "previous_price_signal": None,
            "previous_price_gap": None,
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
        return self._clip(self._safe_float(offer.get("price_signal", 0.5), 0.5), 1e-6, 1.0)

    def _is_invalid_action(self, action: Any) -> bool:
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

        return False

    def compute_step_reward(
        self,
        *,
        action: Any,
        state: Dict[str, Any],
        seller_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        agreement_reached: bool,
        acted_after_agreement: bool,
        step_index: int,
        done: bool,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        rs = state.setdefault("reward_state", self.initialize_reward_state())
        buyer_price = self._price_signal(state)
        counter_offer = seller_out.get("counter_offer", {}) if isinstance(seller_out, dict) else {}
        seller_price = self._safe_float(counter_offer.get("price_signal", buyer_price), buyer_price)

        previous_price = self._safe_float(rs.get("previous_price_signal", buyer_price), buyer_price)
        previous_gap = self._safe_float(
            rs.get("previous_price_gap", abs(previous_price - seller_price)),
            abs(previous_price - seller_price),
        )
        current_gap = abs(buyer_price - seller_price)

        progress_raw = max(previous_gap - current_gap, 0.0)
        fairness_raw = self._clip(1.0 - abs(buyer_price - seller_price), 0.0, 1.0)
        risk_score = self._clip(
            self._safe_float(risk_out.get("risk_score", state.get("risk_score", 0.0)), 0.0),
            0.0,
            1.0,
        )

        deal_closure = 1.0 if agreement_reached else 0.0
        reward = (
            2.0 * float(deal_closure)
            + 1.0 * float(progress_raw)
            + 0.5 * float(fairness_raw)
            - 1.0 * float(risk_score**2)
        )
        if bool(done) and bool(agreement_reached):
            reward = max(float(reward), 1.0)
        reward_unclipped = float(reward)
        reward = self._clip(float(reward), -2.0, 2.5)
        success = 1 if (bool(done) and bool(agreement_reached)) else 0

        components = {
            "deal_closure": float(deal_closure),
            "progress": 1.0 * float(progress_raw),
            "fairness": 0.5 * float(fairness_raw),
            "risk_penalty": -1.0 * float(risk_score**2),
            "success": float(success),
        }

        rs["last_reward"] = float(reward)
        rs["previous_price_signal"] = float(buyer_price)
        rs["previous_price_gap"] = float(current_gap)

        metadata = {
            "total_before_clip": float(reward_unclipped),
            "clip_min": -2.0,
            "clip_max": 2.5,
            "step_index": int(step_index),
            "buyer_price_signal": float(buyer_price),
            "seller_price_signal": float(seller_price),
            "price_gap": float(current_gap),
            "risk_score": float(risk_score),
            "agreement_reached": bool(agreement_reached),
            "done": bool(done),
            "success": int(success),
        }
        return float(reward), components, metadata
