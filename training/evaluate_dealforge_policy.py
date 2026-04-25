from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from training.train_dealforge_grpo_colab import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model id or local model path")
    parser.add_argument("--config", type=str, default="configs/dealforge_default.yaml")
    parser.add_argument("--output", type=str, default="outputs/eval_metrics.json")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    import random

    rng = random.Random(args.seed)
    eval_seeds = [rng.randint(1, 10**9) for _ in range(args.episodes)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    metrics = evaluate_policy(
        model,
        tokenizer,
        env_config_path=args.config,
        eval_seeds=eval_seeds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()
