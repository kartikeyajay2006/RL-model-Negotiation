from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class DealForgeConfig:
    raw: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "DealForgeConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, key: str) -> Dict[str, Any]:
        return self.raw.get(key, {})
