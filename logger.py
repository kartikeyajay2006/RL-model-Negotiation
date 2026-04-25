from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def safe_serialize(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): safe_serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_serialize(v) for v in value]
    return str(value)


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


def flatten_dict(
    nested: Mapping[str, Any],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten nested dictionaries for table-style logging."""
    out: Dict[str, Any] = {}
    for key, value in nested.items():
        key_str = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, parent_key=key_str, sep=sep))
        else:
            out[key_str] = safe_serialize(value)
    return out


def _token_score(name: str, include_tokens: Sequence[str], exclude_tokens: Sequence[str]) -> float:
    lower = name.lower()
    score = 0.0
    for tok in include_tokens:
        if tok.lower() in lower:
            score += 1.0
    for tok in exclude_tokens:
        if tok.lower() in lower:
            score -= 1.0
    return score


def detect_metric_key(
    flat_metrics: Mapping[str, Any],
    *,
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str],
) -> Optional[str]:
    """
    Detect the most likely metric key by token score and numeric convertibility.
    """
    candidates: List[Tuple[float, str]] = []
    for k, v in flat_metrics.items():
        if _safe_float(v) is None:
            continue
        score = _token_score(k, include_tokens, exclude_tokens)
        if score > 0:
            candidates.append((score, k))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # Support JSON array and JSONL.
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [dict(x) for x in arr if isinstance(x, dict)]

    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _coerce_csv_row(row: Mapping[str, Any], fieldnames: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in fieldnames:
        value = row.get(key, "")
        if isinstance(value, (list, tuple, dict)):
            out[key] = json.dumps(safe_serialize(value), ensure_ascii=True)
        else:
            out[key] = safe_serialize(value)
    return out


@dataclass
class Logger:
    run_name: str
    output_root: str | Path = "logs"
    json_filename: str = "training_log.json"
    csv_filename: str = "training_log.csv"
    config_filename: str = "config.json"
    flatten_separator: str = "."

    _records_cache: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _csv_fieldnames: List[str] = field(default_factory=list, init=False)
    _running_reward_sum: float = field(default=0.0, init=False)
    _best_reward: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.run_dir = self.output_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.run_dir / self.json_filename
        self.csv_path = self.run_dir / self.csv_filename
        self.config_path = self.run_dir / self.config_filename

        self._bootstrap_from_existing_logs()

    def save_run_config(self, config_dict: Mapping[str, Any]) -> None:
        self.config_path.write_text(
            json.dumps(safe_serialize(config_dict), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def log_episode(
        self,
        *,
        episode_index: int,
        total_reward: float,
        episode_length: int,
        final_info: Optional[Mapping[str, Any]] = None,
        extra_metrics: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        info = final_info or {}
        extras = extra_metrics or {}

        flat_info = flatten_dict(info, parent_key="info", sep=self.flatten_separator)
        flat_extra = flatten_dict(extras, parent_key="metrics", sep=self.flatten_separator)

        self._running_reward_sum += float(total_reward)
        n_records_next = len(self._records_cache) + 1
        running_avg_reward = self._running_reward_sum / float(max(1, n_records_next))
        self._best_reward = (
            float(total_reward)
            if self._best_reward is None
            else max(self._best_reward, float(total_reward))
        )

        success_key = detect_metric_key(
            flat_info,
            include_tokens=("success", "solved", "win", "goal", "completed"),
            exclude_tokens=("rate", "avg", "mean"),
        )
        risk_key = detect_metric_key(
            flat_info,
            include_tokens=("risk", "unsafe", "hazard", "violation"),
            exclude_tokens=("reward",),
        )
        entropy_key = detect_metric_key(
            {**flat_info, **flat_extra},
            include_tokens=("entropy",),
            exclude_tokens=(),
        )
        constraint_key = detect_metric_key(
            {**flat_info, **flat_extra},
            include_tokens=("constraint", "violation", "breach"),
            exclude_tokens=("risk",),
        )

        safe_length = max(1, int(episode_length))
        record: Dict[str, Any] = {
            "episode_index": int(episode_index),
            "timestamp": float(time.time()),
            "total_reward": float(total_reward),
            "episode_length": int(episode_length),
            "avg_reward_per_step": float(total_reward) / float(safe_length),
            "running_avg_reward": float(running_avg_reward),
            "best_reward": float(self._best_reward),
            "success": _safe_float(flat_info.get(success_key)) if success_key else None,
            "risk_score": _safe_float(flat_info.get(risk_key)) if risk_key else None,
            "entropy": _safe_float((flat_extra if entropy_key in flat_extra else flat_info).get(entropy_key)) if entropy_key else None,
            "constraint_violations": _safe_float((flat_extra if constraint_key in flat_extra else flat_info).get(constraint_key)) if constraint_key else None,
        }
        record.update(flat_info)
        record.update(flat_extra)

        self._append_jsonl(record)
        self._append_csv(record)
        self._records_cache.append(record)
        return record

    def load_records(self) -> List[Dict[str, Any]]:
        return _load_json_records(self.json_path)

    def _bootstrap_from_existing_logs(self) -> None:
        self._records_cache = _load_json_records(self.json_path)
        if self._records_cache:
            rewards = [float(r.get("total_reward", 0.0)) for r in self._records_cache]
            self._running_reward_sum = sum(rewards)
            self._best_reward = max(rewards)
            fieldnames = set()
            for row in self._records_cache:
                fieldnames.update(row.keys())
            self._csv_fieldnames = sorted(fieldnames)

        if self.csv_path.exists() and not self._csv_fieldnames:
            with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self._csv_fieldnames = list(reader.fieldnames or [])

    def _append_jsonl(self, record: Mapping[str, Any]) -> None:
        with self.json_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_serialize(record), ensure_ascii=True) + "\n")

    def _append_csv(self, record: Mapping[str, Any]) -> None:
        incoming_keys = set(record.keys())
        existing_keys = set(self._csv_fieldnames)
        if not self.csv_path.exists():
            self._csv_fieldnames = sorted(incoming_keys)
            self._rewrite_csv(self._records_cache + [dict(record)])
            return

        if not incoming_keys.issubset(existing_keys):
            self._csv_fieldnames = sorted(existing_keys | incoming_keys)
            self._rewrite_csv(self._records_cache + [dict(record)])
            return

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writerow(_coerce_csv_row(record, self._csv_fieldnames))

    def _rewrite_csv(self, rows: Iterable[Mapping[str, Any]]) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(_coerce_csv_row(row, self._csv_fieldnames))

