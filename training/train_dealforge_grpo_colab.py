"""
Colab-ready PPO training pipeline for DealForge (standalone env usage).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from dealforge.openenv_environment import DealForgeMCPEnvironment


ACTION_NAMES = [
    "propose_contract",
    "modify_clause",
    "accept_deal",
    "reject_deal",
    "delay_negotiation",
    "ask_question",
]


@dataclass
class EpisodeResult:
    total_reward: float
    success: bool
    quality: float
    steps: int
    transcript: List[Dict[str, Any]]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(visible_state: Dict[str, Any], horizon: int) -> str:
    return (
        "You are a procurement negotiation agent in DealForge.\n"
        "Return ONLY JSON with this schema:\n"
        "{\"actions\": [{\"action_name\": string, \"payload\": object}, ...]}\n"
        "Rules:\n"
        f"- Use only actions: {ACTION_NAMES}\n"
        f"- Propose 1 to {horizon} actions.\n"
        "- Prefer realistic negotiation, legal safety, and risk-aware decisions.\n"
        "Current visible state:\n"
        f"{json.dumps(visible_state, ensure_ascii=True)}"
    )


def extract_json_block(text: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_actions(obj: Dict[str, Any], max_actions: int) -> List[Dict[str, Any]]:
    actions = obj.get("actions", []) if isinstance(obj, dict) else []
    normalized: List[Dict[str, Any]] = []
    for a in actions[:max_actions]:
        if not isinstance(a, dict):
            continue
        name = a.get("action_name")
        payload = a.get("payload", {})
        if name not in ACTION_NAMES:
            continue
        if not isinstance(payload, dict):
            payload = {}
        normalized.append({"action_name": name, "payload": payload})
    return normalized


def fallback_policy_action(visible_state: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    current = visible_state.get("current_offer", {})
    name = rng.choice(ACTION_NAMES)
    if name == "propose_contract":
        return {"action_name": name, "payload": {"price_signal": float(rng.random())}}
    if name == "modify_clause":
        return {
            "action_name": name,
            "payload": {
                "clause_patch": {
                    "id": f"auto_{rng.randint(1,9999)}",
                    "ambiguity": float(rng.random()),
                    "required_terms_complete": bool(rng.random() > 0.3),
                }
            },
        }
    if name in ("accept_deal", "reject_deal"):
        return {"action_name": name, "payload": {"target_version": current.get("version", 0)}}
    if name == "delay_negotiation":
        return {"action_name": name, "payload": {"delay_hint": int(rng.randint(1, 3))}}
    return {"action_name": "ask_question", "payload": {"target": rng.choice(["seller", "legal", "risk"])}}


def compute_negotiation_quality(transcript: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
    if not transcript:
        return 0.0

    avg_risk = np.mean([float(x.get("risk", {}).get("risk_score", final_state.get("risk_score", 1.0))) for x in transcript])
    avg_legal = np.mean([float(x.get("legal", {}).get("legal_risk", 1.0)) for x in transcript])
    avg_accept_prob = np.mean([float(x.get("seller", {}).get("accept_probability", 0.0)) for x in transcript])
    final_trust = float(final_state.get("trust_score", 0.0))
    final_pressure = float(final_state.get("external_pressure", 1.0))

    quality = (
        0.28 * avg_accept_prob
        + 0.24 * final_trust
        + 0.16 * (1.0 - final_pressure)
        + 0.16 * (1.0 - avg_risk)
        + 0.16 * (1.0 - avg_legal)
    )
    return float(max(0.0, min(1.0, quality)))


def rollout_episode_from_actions(
    env_config_path: str,
    actions: List[Dict[str, Any]],
    seed: int,
) -> EpisodeResult:
    env = DealForgeMCPEnvironment(config=env_config_path, seed=seed)
    obs = env.reset(seed=seed)

    transcript: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = False

    for i, a in enumerate(actions):
        if done:
            break
        obs, reward, done, info = env.step(a)
        total_reward += float(reward or 0.0)

        event_log = info.get("event_log", []) if isinstance(info, dict) else []
        step_event = event_log[-1] if event_log else {}
        step_event["step"] = i + 1
        transcript.append(step_event)

    rng = random.Random(seed + 17)
    while not done:
        a = fallback_policy_action(obs, rng)
        obs, reward, done, info = env.step(a)
        total_reward += float(reward or 0.0)

        event_log = info.get("event_log", []) if isinstance(info, dict) else []
        step_event = event_log[-1] if event_log else {}
        step_event["step"] = len(transcript) + 1
        transcript.append(step_event)

    st = env.state().get("dynamic_state", {})
    terminal = st.get("terminal", {})
    success = bool(terminal.get("deal_success", False))
    quality = compute_negotiation_quality(transcript, st)
    steps = int(st.get("time_steps", {}).get("current", len(transcript)))

    env.close()
    return EpisodeResult(total_reward=total_reward, success=success, quality=quality, steps=steps, transcript=transcript)


def generate_plan_with_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def evaluate_policy(
    model,
    tokenizer,
    env_config_path: str,
    eval_seeds: List[int],
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    rewards, successes, qualities, steps = [], [], [], []
    transcripts = []

    for s in eval_seeds:
        env = DealForgeMCPEnvironment(config=env_config_path, seed=s)
        obs = env.reset(seed=s)

        prompt = build_prompt(obs, horizon=obs.get("time_steps", {}).get("max", 20))
        completion = generate_plan_with_model(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        parsed = extract_json_block(completion) or {}
        actions = normalize_actions(parsed, max_actions=30)
        if not actions:
            rng = random.Random(s + 123)
            actions = [fallback_policy_action(obs, rng) for _ in range(3)]

        env.close()

        result = rollout_episode_from_actions(env_config_path, actions, seed=s)
        rewards.append(result.total_reward)
        successes.append(1 if result.success else 0)
        qualities.append(result.quality)
        steps.append(result.steps)
        transcripts.append(
            {
                "seed": s,
                "prompt": prompt,
                "model_completion": completion,
                "actions": actions,
                "episode": {
                    "total_reward": result.total_reward,
                    "success": result.success,
                    "quality": result.quality,
                    "steps": result.steps,
                },
                "transcript": result.transcript,
            }
        )

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "quality_mean": float(np.mean(qualities)),
        "steps_mean": float(np.mean(steps)),
        "rewards": rewards,
        "successes": successes,
        "qualities": qualities,
        "steps": steps,
        "transcripts": transcripts,
    }


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def plot_comparison(before: Dict[str, Any], after: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    x = list(range(len(before["rewards"])))
    plt.plot(x, before["rewards"], label="baseline", linewidth=1.8)
    plt.plot(x, after["rewards"], label="trained", linewidth=1.8)
    plt.title("DealForge Reward Over Evaluation Episodes")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Episode Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_over_time.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    labels = ["Success Rate", "Quality Mean"]
    baseline_vals = [before["success_rate"], before["quality_mean"]]
    trained_vals = [after["success_rate"], after["quality_mean"]]
    idx = np.arange(len(labels))
    w = 0.35
    plt.bar(idx - w / 2, baseline_vals, w, label="baseline")
    plt.bar(idx + w / 2, trained_vals, w, label="trained")
    plt.xticks(idx, labels)
    plt.ylim(0.0, 1.0)
    plt.title("Before vs After: Negotiation Outcomes")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_quality_comparison.png", dpi=150)
    plt.close(fig)


def create_training_dataset(config_path: str, n_prompts: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_prompts):
        s = rng.randint(1, 10**9)
        env = DealForgeMCPEnvironment(config=config_path, seed=s)
        obs = env.reset(seed=s)
        horizon = int(obs.get("time_steps", {}).get("max", 20))
        prompt = build_prompt(obs, horizon=horizon)
        env.close()

        rows.append({"prompt": prompt, "response": "", "reward": 0.0, "seed": s})
    return Dataset.from_list(rows)


def write_summary(out_dir: Path, before: Dict[str, Any], after: Dict[str, Any]) -> None:
    lines = []
    lines.append("# DealForge PPO Summary")
    lines.append("")
    lines.append("## Baseline")
    lines.append(f"- reward_mean: {before['reward_mean']:.4f}")
    lines.append(f"- success_rate: {before['success_rate']:.4f}")
    lines.append(f"- quality_mean: {before['quality_mean']:.4f}")
    lines.append("")
    lines.append("## Trained")
    lines.append(f"- reward_mean: {after['reward_mean']:.4f}")
    lines.append(f"- success_rate: {after['success_rate']:.4f}")
    lines.append(f"- quality_mean: {after['quality_mean']:.4f}")
    lines.append("")
    lines.append("## Delta (trained - baseline)")
    lines.append(f"- reward_mean_delta: {after['reward_mean'] - before['reward_mean']:.4f}")
    lines.append(f"- success_rate_delta: {after['success_rate'] - before['success_rate']:.4f}")
    lines.append(f"- quality_mean_delta: {after['quality_mean'] - before['quality_mean']:.4f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def ppo_train_with_environment(
    *,
    trainer: PPOTrainer,
    train_dataset: Dataset,
    tokenizer,
    model,
    env_config_path: str,
    max_steps: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    rng = random.Random(12345)
    logs: List[Dict[str, Any]] = []

    device = torch.device("cpu")

    prompts = list(train_dataset["prompt"])
    if not prompts:
        raise ValueError("Training dataset is empty.")

    for step in range(max_steps):
        batch_prompts = [prompts[rng.randrange(len(prompts))] for _ in range(batch_size)]

        query_tensors: List[torch.Tensor] = []
        response_tensors: List[torch.Tensor] = []
        rewards_list: List[float] = []

        for p in batch_prompts:
            query_ids = tokenizer(p, return_tensors="pt").input_ids[0].to(device)

            with torch.no_grad():
                generation = model.generate(
                    input_ids=query_ids.unsqueeze(0),
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]

            response_ids = generation[query_ids.shape[0] :]
            if response_ids.shape[0] == 0:
                response_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
            response_ids = response_ids.to(device)
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            parsed = extract_json_block(response_text) or {}
            actions = normalize_actions(parsed, max_actions=30)
            if not actions:
                actions = [fallback_policy_action({}, random.Random(step + 777)) for _ in range(3)]

            rollout_seed = rng.randint(1, 10**9)
            ep = rollout_episode_from_actions(env_config_path, actions, rollout_seed)

            query_tensors.append(query_ids)
            response_tensors.append(response_ids)
            rewards_list.append(float(ep.total_reward))

        reward_tensors = [
            torch.tensor(r, dtype=torch.float32).to(device)
            for r in rewards_list
        ]

        if len(query_tensors) > 0 and len(response_tensors) > 0:
            stats = trainer.step(query_tensors, response_tensors, reward_tensors)
        else:
            continue

        step_log = {
            "step": step,
            "reward_mean": float(np.mean(rewards_list)),
            "reward_std": float(np.std(rewards_list)),
            "stats": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in stats.items()},
        }
        logs.append(step_log)
        print(f"Step {step} | reward_mean={np.mean(rewards_list):.4f}")

    return {"logs": logs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dealforge_default.yaml")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="outputs/dealforge_ppo_run")
    parser.add_argument("--train_prompts", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{run_name}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ = load_config(args.config)

    train_ds = create_training_dataset(args.config, args.train_prompts, args.seed)

    eval_rng = random.Random(args.seed + 1001)
    eval_seeds = [eval_rng.randint(1, 10**9) for _ in range(args.eval_episodes)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    baseline_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    baseline_metrics = evaluate_policy(
        baseline_model,
        tokenizer,
        env_config_path=args.config,
        eval_seeds=eval_seeds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    save_json(out_dir / "baseline_metrics.json", baseline_metrics)
    del baseline_model

    ppo_config = PPOConfig(
        model_name=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        log_with=None,
        seed=args.seed,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    device = torch.device("cpu")
    model.to(device)
    ref_model.to(device)
    print(type(model))

    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=ppo_config,
        tokenizer=tokenizer,
    )

    train_output = ppo_train_with_environment(
        trainer=trainer,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        model=model,
        env_config_path=args.config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    trained_model_dir = out_dir / "trained_model"
    trained_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(trained_model_dir)
    tokenizer.save_pretrained(trained_model_dir)

    save_json(out_dir / "train_output.json", train_output)

    trained_model = AutoModelForCausalLMWithValueHead.from_pretrained(str(trained_model_dir))
    trained_metrics = evaluate_policy(
        trained_model,
        tokenizer,
        env_config_path=args.config,
        eval_seeds=eval_seeds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    save_json(out_dir / "trained_metrics.json", trained_metrics)

    k = min(8, len(eval_seeds))
    save_jsonl(out_dir / "transcripts_baseline.jsonl", baseline_metrics["transcripts"][:k])
    save_jsonl(out_dir / "transcripts_trained.jsonl", trained_metrics["transcripts"][:k])

    comparison_log = {
        "baseline": {
            "reward_mean": baseline_metrics["reward_mean"],
            "success_rate": baseline_metrics["success_rate"],
            "quality_mean": baseline_metrics["quality_mean"],
            "steps_mean": baseline_metrics["steps_mean"],
        },
        "trained": {
            "reward_mean": trained_metrics["reward_mean"],
            "success_rate": trained_metrics["success_rate"],
            "quality_mean": trained_metrics["quality_mean"],
            "steps_mean": trained_metrics["steps_mean"],
        },
        "delta": {
            "reward_mean": trained_metrics["reward_mean"] - baseline_metrics["reward_mean"],
            "success_rate": trained_metrics["success_rate"] - baseline_metrics["success_rate"],
            "quality_mean": trained_metrics["quality_mean"] - baseline_metrics["quality_mean"],
        },
    }
    save_json(out_dir / "comparison.json", comparison_log)

    plot_comparison(baseline_metrics, trained_metrics, out_dir)
    write_summary(out_dir, baseline_metrics, trained_metrics)

    print("Training + evaluation complete.")
    print(f"Artifacts saved to: {out_dir}")
    print("Key files:")
    print("- baseline_metrics.json")
    print("- trained_metrics.json")
    print("- comparison.json")
    print("- reward_over_time.png")
    print("- success_quality_comparison.png")
    print("- transcripts_baseline.jsonl")
    print("- transcripts_trained.jsonl")


if __name__ == "__main__":
    main()
