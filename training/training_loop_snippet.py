from __future__ import annotations

from typing import Any, Dict, Protocol

from logger import Logger


class AgentProtocol(Protocol):
    def predict(self, state: Dict[str, Any]) -> Any:
        ...


def run_training_loop(
    *,
    env: Any,
    agent: AgentProtocol,
    logger: Logger,
    num_episodes: int,
) -> None:
    """
    Generic RL training loop integration.

    Assumes:
    - state, reset_info = env.reset(...)
    - step_out = env.step(action)
      where step_out is either:
      - tuple: (next_state, reward, done, info), or
      - object with attrs: observation/reward/terminated/truncated/info
    """
    for episode_index in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        final_info: Dict[str, Any] = {}

        while not done:
            action = agent.predict(state)
            step_out = env.step(action)

            if hasattr(step_out, "observation"):
                state = step_out.observation
                reward = float(step_out.reward)
                done = bool(step_out.terminated or step_out.truncated)
                final_info = dict(step_out.info) if isinstance(step_out.info, dict) else {}
            else:
                next_state, reward, done_flag, info = step_out
                state = next_state
                reward = float(reward)
                done = bool(done_flag)
                final_info = dict(info) if isinstance(info, dict) else {}

            total_reward += reward
            steps += 1

        logger.log_episode(
            episode_index=episode_index,
            total_reward=total_reward,
            episode_length=steps,
            final_info=final_info,
        )

