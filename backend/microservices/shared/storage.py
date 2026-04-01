from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Optional


BASE_STORAGE_DIR = Path("backend") / "microservices" / "storage"
DATASET_DIR = BASE_STORAGE_DIR / "datasets"
CHECKPOINT_DIR = BASE_STORAGE_DIR / "checkpoints"


def ensure_storage_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_dataset_path(dataset_id: int) -> Path:
    ensure_storage_dirs()
    return DATASET_DIR / f"dataset_{int(dataset_id)}.json.gz"


def _safe_checkpoint_path(checkpoint_name: str) -> Path:
    ensure_storage_dirs()
    sanitized = checkpoint_name.replace("/", "_").replace("\\", "_")
    if not sanitized.endswith(".pth"):
        sanitized += ".pth"
    return CHECKPOINT_DIR / sanitized


def save_dataset_records(dataset_id: int, records: list[dict[str, Any]]) -> str:
    path = _safe_dataset_path(dataset_id)
    payload = json.dumps(records, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return str(path.as_posix())


def load_dataset_records(file_path: Optional[str]) -> list[dict[str, Any]]:
    if not file_path:
        return []
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return []
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, list) else []


def save_checkpoint_binary(checkpoint_name: str, payload: bytes) -> str:
    path = _safe_checkpoint_path(checkpoint_name)
    path.write_bytes(payload)
    return str(path.as_posix())


def load_checkpoint_binary(file_path: str) -> bytes:
    path = Path(file_path)
    return path.read_bytes()
