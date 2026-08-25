"""config.yaml을 읽어 파이썬 값으로 내어준다. 숫자는 전부 여기를 거쳐서 읽는다 (CLAUDE.md 참고)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    # config.yaml을 한 번만 읽어 캐시해서 돌려준다
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(*keys: str, default: Any = None) -> Any:
    # 'exposure', 'enter' 처럼 키를 순서대로 넘겨 중첩된 값을 꺼낸다
    value: Any = load_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
