from __future__ import annotations

import math
from typing import Any


def sanitize_float(value: Any) -> Any:
    """Заменяет NaN/Inf float значения на None для безопасной JSON-сериализации."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def sanitize_data(data: Any) -> Any:
    """Рекурсивно заменяет NaN/Inf float значения на None во вложенных dict/list."""
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data
