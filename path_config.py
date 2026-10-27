from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def configured_data_root() -> Path:
    configured = os.environ.get("TFI_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "data"


DATA_ROOT = configured_data_root()
RESULT_ROOT = PROJECT_ROOT / "result"

CLEAN_DATASET = "D_domainnet_clean"
NIGHTSHADE_DATASET = "D_domainnet_nightshade_linf_004"
GAUSSIAN_DATASET = "D_domainnet_gaussian_linf_004"
GAUSSIAN_LPIPS_DATASET = "D_domainnet_gaussian_lpips_matched_ns_linf_004"

DOMAINNET_DATASETS = (
    CLEAN_DATASET,
    NIGHTSHADE_DATASET,
    GAUSSIAN_DATASET,
    GAUSSIAN_LPIPS_DATASET,
)


def store_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_stored_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
