from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, Protocol

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dealforge.config import DealForgeConfig
from dealforge.env import DealForgeEnv
from logger import Logger
from plot import (
    auto_detect_keys,
    compute_training_summary,
    load_training_json,
    print_training_summary,
    render_multi_panel_plot,
    save_summary,
)


class AgentProtocol(Protocol):
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...


class RandomAgent:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        actions = state.get("available_actions", [])
        if not actions:
            return {
                "action_type": "ask_question",
                "payload": {"target_agent": "seller", "question_type": "pricing"},
            }
        action_type = self.rng.choice(actions)
        contract = state.get("current_contract", {})
        if action_type == "propose_contract":
            return {
                "action_type": action_type,
                "payload": {"contract_version_base": contract.get("version_id")},
            }
        if action_type == "modify_clause":
            return {
                "action_type": action_type,
                "payload": {
                    "operation": self.rng.choice(["add", "replace", "remove"]),
                    "target_clause_id": "unknown",
                    "new_text_or_struct": {"value": self.rng.random()},
                },
            }
        if action_type in ("accept_deal", "reject_deal"):
            return {
                "action_type": action_type,
                "payload": {"target_contract_version": contract.get("version_id")},
            }
        if action_type == "delay_negotiation":
            return {
                "action_type": action_type,
                "payload": {"requested_duration": self.rng.randint(1, 2)},
            }
        return {
            "action_type": "ask_question",
            "payload": {
                "target_agent": self.rng.choice(["seller", "legal", "risk"]),
                "question_type": self.rng.choice(
                    ["pricing", "liability", "timeline", "compliance"]
                ),
            },
        }


def generic_training_loop(
    *,
    env: Any,
    agent: AgentProtocol,
    logger: Logger,
    num_episodes: int,
    seed_start: int,
) -> None:
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed_start + episode)
        done = False
        total_reward = 0.0
        steps = 0
        final_info: Dict[str, Any] = {}

        while not done:
            action = agent.predict(state)
            step_out = env.step(action)
            state = step_out.observation
            total_reward += float(step_out.reward)
            steps += 1
            done = bool(step_out.terminated or step_out.truncated)
            final_info = dict(step_out.info) if isinstance(step_out.info, dict) else {}

        logger.log_episode(
            episode_index=episode,
            total_reward=total_reward,
            episode_length=steps,
            final_info=final_info,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dealforge_default.yaml")
    parser.add_argument("--run_name", type=str, default="dealforge_demo")
    parser.add_argument("--output_root", type=str, default="logs")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--baseline_window", type=int, default=10)
    parser.add_argument("--summary_window", type=int, default=10)
    parser.add_argument("--learning_phase_ratio", type=float, default=0.3)
    args = parser.parse_args()

    cfg = DealForgeConfig.from_yaml(args.config)
    env = DealForgeEnv(cfg, seed=args.seed)
    agent = RandomAgent(seed=args.seed)
    logger = Logger(run_name=args.run_name, output_root=args.output_root)
    logger.save_run_config(
        {
            "config_path": args.config,
            "episodes": args.episodes,
            "seed": args.seed,
            "smooth_window": args.smooth_window,
            "baseline_window": args.baseline_window,
            "summary_window": args.summary_window,
            "learning_phase_ratio": args.learning_phase_ratio,
        }
    )

    generic_training_loop(
        env=env,
        agent=agent,
        logger=logger,
        num_episodes=args.episodes,
        seed_start=args.seed,
    )

    run_dir = Path(args.output_root) / args.run_name
    records = load_training_json(run_dir / "training_log.json")
    keys = auto_detect_keys(records)
    render_multi_panel_plot(
        records,
        output_png=run_dir / "reward_curve.png",
        output_pdf=run_dir / "reward_curve.pdf",
        reward_key=str(keys["reward_key"]),
        success_key=keys["success_key"],
        risk_key=keys["risk_key"],
        smooth_window=max(1, int(args.smooth_window)),
        baseline_window=max(1, int(args.baseline_window)),
        learning_phase_ratio=max(0.0, min(1.0, float(args.learning_phase_ratio))),
        title="DealForge Training Diagnostics",
    )

    summary = compute_training_summary(
        records,
        reward_key=str(keys["reward_key"]),
        success_key=keys["success_key"],
        summary_window=max(1, int(args.summary_window)),
    )
    save_summary(run_dir / "summary.json", summary)
    print_training_summary(summary)

    print(f"Saved run artifacts in: {run_dir}")


if __name__ == "__main__":
    main()

