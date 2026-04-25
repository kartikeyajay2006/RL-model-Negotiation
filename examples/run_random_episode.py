from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dealforge.config import DealForgeConfig
from dealforge.env import DealForgeEnv


def random_action(obs):
    a = random.choice(obs["available_actions"])
    current = obs["current_contract"]["version_id"]
    if a == "propose_contract":
        return {"action_type": a, "payload": {"contract_version_base": current}}
    if a == "modify_clause":
        return {
            "action_type": a,
            "payload": {
                "operation": random.choice(["add", "replace", "remove"]),
                "target_clause_id": "unknown",
                "new_text_or_struct": {"param": random.random()},
            },
        }
    if a == "accept_deal":
        return {"action_type": a, "payload": {"target_contract_version": current}}
    if a == "reject_deal":
        return {"action_type": a, "payload": {"target_contract_version": current}}
    if a == "delay_negotiation":
        return {"action_type": a, "payload": {"requested_duration": random.randint(1, 2)}}
    if a == "ask_question":
        return {
            "action_type": a,
            "payload": {
                "target_agent": random.choice(["seller", "legal", "risk"]),
                "question_type": random.choice(["pricing", "liability", "timeline", "compliance"]),
            },
        }
    raise ValueError(a)


def main():
    cfg = DealForgeConfig.from_yaml("configs/dealforge_default.yaml")
    env = DealForgeEnv(cfg, seed=7)

    obs, info = env.reset()
    print("reset", info)

    done = False
    total = 0.0
    while not done:
        act = random_action(obs)
        out = env.step(act)
        total += out.reward
        done = out.terminated or out.truncated
        obs = out.observation
        print(
            f"step={obs['step']} action={act['action_type']} reward={out.reward:.3f} "
            f"term={out.terminated} trunc={out.truncated}"
        )

    print(f"episode_done total_reward={total:.3f}")


if __name__ == "__main__":
    main()
