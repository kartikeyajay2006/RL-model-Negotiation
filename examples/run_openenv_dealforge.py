from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dealforge.openenv_environment import DealForgeAction, DealForgeMCPEnvironment


def sample_action(obs):
    action_name = random.choice(obs.visible_state["available_actions"])
    offer = obs.visible_state["current_offer"]

    if action_name == "propose_contract":
        return DealForgeAction(action_name=action_name, payload={"price_signal": random.random()})
    if action_name == "modify_clause":
        return DealForgeAction(
            action_name=action_name,
            payload={
                "clause_patch": {
                    "id": f"c_{random.randint(1,999)}",
                    "ambiguity": random.random(),
                    "required_terms_complete": random.choice([True, False]),
                }
            },
        )
    if action_name == "accept_deal":
        return DealForgeAction(action_name=action_name, payload={"target_version": offer.get("version")})
    if action_name == "reject_deal":
        return DealForgeAction(action_name=action_name, payload={"target_version": offer.get("version")})
    if action_name == "delay_negotiation":
        return DealForgeAction(action_name=action_name, payload={"delay_hint": random.randint(1, 3)})
    return DealForgeAction(action_name="ask_question", payload={"target": random.choice(["seller", "legal", "risk"])})


def main():
    env = DealForgeMCPEnvironment(config="configs/dealforge_default.yaml", seed=123)

    obs = env.reset(seed=123)
    print("reset", obs.visible_state["time_steps"])

    done = False
    while not done:
        action = sample_action(obs)
        obs = env.step(action)
        print(
            f"step={obs.visible_state['time_steps']['current']} action={action.action_name} "
            f"reward={obs.reward:.3f} done={obs.done}"
        )
        done = obs.done

    full_state = env.state
    print("terminal", full_state.dynamic_state.get("terminal"))
    env.close()


if __name__ == "__main__":
    main()
