from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def load_training_json(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log file not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [x for x in data if isinstance(x, dict)]

    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def flatten_dict(
    nested: Mapping[str, Any],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in nested.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, parent_key=key, sep=sep))
        else:
            out[key] = v
    return out


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "y", "success", "passed"}:
            return 1.0
        if low in {"false", "no", "n", "failure", "failed"}:
            return 0.0
        try:
            f = float(low)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except ValueError:
            return None
    return None


def moving_average(values: Sequence[float], window_size: int) -> List[float]:
    if window_size <= 1:
        return [float(v) for v in values]
    out: List[float] = []
    q: List[float] = []
    running_sum = 0.0
    for v in values:
        q.append(float(v))
        running_sum += float(v)
        if len(q) > window_size:
            running_sum -= q.pop(0)
        out.append(running_sum / float(len(q)))
    return out


def rolling_std(values: Sequence[float], window_size: int) -> List[float]:
    if window_size <= 1:
        return [0.0] * len(values)
    out: List[float] = []
    q: List[float] = []
    for v in values:
        q.append(float(v))
        if len(q) > window_size:
            q.pop(0)
        m = sum(q) / float(len(q))
        var = sum((x - m) ** 2 for x in q) / float(len(q))
        out.append(math.sqrt(var))
    return out


def _token_score(name: str, include_tokens: Sequence[str], exclude_tokens: Sequence[str]) -> float:
    low = name.lower()
    score = 0.0
    for t in include_tokens:
        if t.lower() in low:
            score += 1.0
    for t in exclude_tokens:
        if t.lower() in low:
            score -= 1.0
    return score


def detect_metric_key(
    records: Sequence[Dict[str, Any]],
    *,
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str],
    preferred_keys: Sequence[str] = (),
    min_presence_ratio: float = 0.2,
) -> Optional[str]:
    if not records:
        return None

    n = len(records)
    counts: Dict[str, int] = {}
    values: Dict[str, List[float]] = {}
    for r in records:
        flat = flatten_dict(r)
        for k, v in flat.items():
            f = _safe_float(v)
            if f is None:
                continue
            counts[k] = counts.get(k, 0) + 1
            values.setdefault(k, []).append(f)

    for k in preferred_keys:
        if k in counts:
            return k

    scored: List[Tuple[float, str]] = []
    for k, cnt in counts.items():
        ratio = cnt / float(max(1, n))
        if ratio < min_presence_ratio:
            continue
        token_score = _token_score(k, include_tokens, exclude_tokens)
        if token_score <= 0:
            continue
        series = values.get(k, [])
        unique = len(set(round(x, 8) for x in series))
        variability = min(1.0, unique / 10.0)
        score = token_score + 0.4 * ratio + 0.3 * variability
        scored.append((score, k))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def auto_detect_keys(
    records: Sequence[Dict[str, Any]],
    *,
    reward_hint: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    reward_key = reward_hint
    if reward_key is None:
        reward_key = detect_metric_key(
            records,
            include_tokens=("reward", "return", "score"),
            exclude_tokens=("risk", "loss", "violation"),
            preferred_keys=("total_reward", "reward", "episode_reward"),
        )
    if reward_key is None:
        reward_key = "total_reward"

    success_key = detect_metric_key(
        records,
        include_tokens=("success", "solved", "win", "goal", "completed"),
        exclude_tokens=("rate", "avg", "mean"),
        preferred_keys=("success", "info.success", "metrics.success"),
    )

    risk_key = detect_metric_key(
        records,
        include_tokens=("risk", "unsafe", "hazard", "violation"),
        exclude_tokens=("reward", "return"),
        preferred_keys=("risk_score", "info.risk_score", "metrics.risk_score"),
    )

    return {
        "reward_key": reward_key,
        "success_key": success_key,
        "risk_key": risk_key,
    }


def extract_series(
    records: Sequence[Dict[str, Any]],
    *,
    key: Optional[str],
    default_value: float = 0.0,
) -> List[float]:
    if key is None:
        return [default_value for _ in records]
    out: List[float] = []
    for r in records:
        flat = flatten_dict(r)
        f = _safe_float(flat.get(key))
        out.append(default_value if f is None else f)
    return out


def _window_split(values: Sequence[float], n: int) -> Tuple[List[float], List[float]]:
    n_used = min(len(values), max(1, int(n)))
    return list(values[:n_used]), list(values[-n_used:])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(max(1, len(values)))


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / float(len(values))


def _pct_change(first: float, last: float) -> float:
    if abs(first) < 1e-12:
        return float("inf") if last > 0 else 0.0
    return ((last - first) / abs(first)) * 100.0


def compute_training_summary(
    records: Sequence[Dict[str, Any]],
    *,
    reward_key: str,
    success_key: Optional[str],
    summary_window: int,
) -> Dict[str, float]:
    rewards = extract_series(records, key=reward_key, default_value=0.0)
    first_r, last_r = _window_split(rewards, summary_window)
    reward_improvement = _pct_change(_mean(first_r), _mean(last_r))

    success_improvement = 0.0
    if success_key is not None:
        success = extract_series(records, key=success_key, default_value=0.0)
        first_s, last_s = _window_split(success, summary_window)
        success_improvement = _pct_change(_mean(first_s), _mean(last_s))

    var_first = _variance(first_r)
    var_last = _variance(last_r)
    variance_reduction = ((_pct_change(var_first, var_last)) * -1.0) if var_first != 0 else 0.0

    return {
        "reward_improvement_pct": reward_improvement,
        "success_improvement_pct": success_improvement,
        "variance_reduction_pct": variance_reduction,
        "first_reward_mean": _mean(first_r),
        "last_reward_mean": _mean(last_r),
        "first_reward_variance": var_first,
        "last_reward_variance": var_last,
        "summary_window": float(min(len(rewards), max(1, summary_window))),
    }


def print_training_summary(summary: Mapping[str, float]) -> None:
    print("===== TRAINING SUMMARY =====")
    if math.isinf(summary["reward_improvement_pct"]):
        print("Reward Improvement: inf%")
    else:
        print(f"Reward Improvement: {summary['reward_improvement_pct']:.2f}%")
    if math.isinf(summary["success_improvement_pct"]):
        print("Success Rate Improvement: inf%")
    else:
        print(f"Success Rate Improvement: {summary['success_improvement_pct']:.2f}%")
    if math.isinf(summary["variance_reduction_pct"]):
        print("Variance Reduction: inf%")
    else:
        print(f"Variance Reduction: {summary['variance_reduction_pct']:.2f}%")


def render_multi_panel_plot(
    records: Sequence[Dict[str, Any]],
    *,
    output_png: str | Path,
    output_pdf: str | Path,
    reward_key: str,
    success_key: Optional[str],
    risk_key: Optional[str],
    smooth_window: int,
    baseline_window: int,
    learning_phase_ratio: float,
    title: str,
) -> None:
    if not records:
        raise ValueError("No records to plot.")

    episodes = [int(flatten_dict(r).get("episode_index", i)) for i, r in enumerate(records)]
    rewards = extract_series(records, key=reward_key, default_value=0.0)
    reward_ma = moving_average(rewards, smooth_window)
    reward_std = rolling_std(rewards, smooth_window)

    baseline_n = min(len(rewards), max(1, int(baseline_window)))
    baseline = _mean(rewards[:baseline_n])

    success_raw = extract_series(records, key=success_key, default_value=0.0) if success_key else [0.0] * len(records)
    success_ma = moving_average(success_raw, smooth_window)

    risk_raw = extract_series(records, key=risk_key, default_value=0.0) if risk_key else [0.0] * len(records)
    risk_ma = moving_average(risk_raw, smooth_window)

    learn_idx = int(max(0, min(len(episodes) - 1, round((len(episodes) - 1) * learning_phase_ratio))))
    final_x = episodes[-1]
    final_y = reward_ma[-1]
    improvement = final_y - baseline

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(title, fontsize=14, y=0.995)

    ax1, ax2, ax3 = axes

    # Panel 1: Reward
    ax1.plot(episodes, rewards, label="Raw Reward", alpha=0.28, linewidth=1.2, color="#1f77b4")
    ax1.plot(episodes, reward_ma, label=f"Moving Avg (w={smooth_window})", linewidth=2.0, color="#0d3b66")
    lower = [m - s for m, s in zip(reward_ma, reward_std)]
    upper = [m + s for m, s in zip(reward_ma, reward_std)]
    ax1.fill_between(episodes, lower, upper, alpha=0.18, color="#4f8bc9", label="Variance Band (±1 std)")
    ax1.axhline(baseline, linestyle="--", linewidth=1.4, color="#9a031e", label=f"Baseline (first {baseline_n})")
    ax1.axvline(episodes[learn_idx], linestyle=":", linewidth=1.4, color="#8338ec", label="Learning Phase Marker")
    ax1.scatter([final_x], [final_y], color="#06d6a0", s=60, zorder=5, label="Final Performance")
    ax1.annotate(
        f"Δ {improvement:.3f}",
        xy=(final_x, final_y),
        xytext=(episodes[max(0, learn_idx // 2)], baseline),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#06d6a0"},
        fontsize=10,
    )
    ax1.set_title("Reward vs Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # Panel 2: Success
    if success_key is not None:
        ax2.plot(episodes, success_raw, label=f"Raw Success ({success_key})", alpha=0.3, linewidth=1.2, color="#2a9d8f")
        ax2.plot(episodes, success_ma, label=f"Smoothed Success (w={smooth_window})", linewidth=2.0, color="#006d77")
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No success metric detected", transform=ax2.transAxes, ha="center", va="center")
    ax2.set_title("Success Rate vs Episode")
    ax2.set_ylabel("Success")
    ax2.grid(True, alpha=0.25)

    # Panel 3: Risk
    if risk_key is not None:
        ax3.plot(episodes, risk_raw, label=f"Raw Risk ({risk_key})", alpha=0.28, linewidth=1.2, color="#f77f00")
        ax3.plot(episodes, risk_ma, label=f"Smoothed Risk (w={smooth_window})", linewidth=2.0, color="#d62828")
        ax3.legend(loc="best")
    else:
        ax3.text(0.5, 0.5, "No risk metric detected", transform=ax3.transAxes, ha="center", va="center")
    ax3.set_title("Risk vs Episode (Lower is better)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Risk")
    ax3.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.975])

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.savefig(output_pdf)
    plt.close(fig)


def save_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="logs")
    parser.add_argument("--json_filename", type=str, default="training_log.json")
    parser.add_argument("--png_filename", type=str, default="reward_curve.png")
    parser.add_argument("--pdf_filename", type=str, default="reward_curve.pdf")
    parser.add_argument("--summary_filename", type=str, default="summary.json")
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--baseline_window", type=int, default=10)
    parser.add_argument("--summary_window", type=int, default=10)
    parser.add_argument("--learning_phase_ratio", type=float, default=0.3)
    parser.add_argument("--reward_key", type=str, default=None)
    parser.add_argument("--title", type=str, default="DealForge RL Training Diagnostics")
    args = parser.parse_args()

    run_dir = Path(args.output_root) / args.run_name
    records = load_training_json(run_dir / args.json_filename)
    if not records:
        raise ValueError(f"No log records found in {run_dir / args.json_filename}")

    keys = auto_detect_keys(records, reward_hint=args.reward_key)
    reward_key = str(keys["reward_key"])
    success_key = keys["success_key"]
    risk_key = keys["risk_key"]

    render_multi_panel_plot(
        records,
        output_png=run_dir / args.png_filename,
        output_pdf=run_dir / args.pdf_filename,
        reward_key=reward_key,
        success_key=success_key,
        risk_key=risk_key,
        smooth_window=max(1, int(args.smooth_window)),
        baseline_window=max(1, int(args.baseline_window)),
        learning_phase_ratio=max(0.0, min(1.0, float(args.learning_phase_ratio))),
        title=args.title,
    )

    summary = compute_training_summary(
        records,
        reward_key=reward_key,
        success_key=success_key,
        summary_window=max(1, int(args.summary_window)),
    )
    summary_with_keys = {
        **summary,
        "detected_reward_key": reward_key,
        "detected_success_key": success_key,
        "detected_risk_key": risk_key,
    }
    save_summary(run_dir / args.summary_filename, summary_with_keys)
    print_training_summary(summary_with_keys)

    print(f"Saved plot: {run_dir / args.png_filename}")
    print(f"Saved plot: {run_dir / args.pdf_filename}")
    print(f"Saved summary: {run_dir / args.summary_filename}")


if __name__ == "__main__":
    main()
