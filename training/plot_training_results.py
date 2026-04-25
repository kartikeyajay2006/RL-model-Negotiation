from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.train_dealforge_grpo_colab import plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline_metrics.json")
    parser.add_argument("--trained", type=str, required=True, help="Path to trained_metrics.json")
    parser.add_argument("--out_dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    with open(args.baseline, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(args.trained, "r", encoding="utf-8") as f:
        after = json.load(f)

    out_dir = Path(args.out_dir)
    plot_comparison(before, after, out_dir)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
