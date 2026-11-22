"""Calculate SD VAE--JPEG Q85 TFI for 14 clean-image processing histories."""
from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Hardcoded configuration
# =========================================================
ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = ROOT / "result" / "features"
FEATURE_METADATA = FEATURE_ROOT / "feature_metadata.parquet"
FEATURES_NPY = FEATURE_ROOT / "sd_vae" / "features.float16.npy"
OUTPUT_CSV = ROOT / "result" / "tfi" / "010_benign_tfi_sd_vae_jpeg85.csv"
BATCH_SIZE = 512
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
EXCLUDED_CLIPART = {14, 32, 79, 123, 465}
VARIANTS = (
    *(("downscaling", str(value)) for value in (0.75, 0.6, 0.5, 0.4, 0.3, 0.25)),
    *(("gaussian_blur", str(value)) for value in (0.5, 1.0, 2.0, 3.0)),
    *(("jpeg_comp", str(value)) for value in (95, 85, 75, 50)),
)
OUTPUT_COLUMNS = [
    "domain", "data_index", "benign_transformation", "benign_factor",
    "latent_distance",
]


def source_rows(data: pd.DataFrame) -> pd.DataFrame:
    result = data.loc[data["domain"].isin(DOMAINS)].copy()
    indices = pd.to_numeric(result["data_index"])
    result["data_index"] = indices.map(lambda value: f"{int(value):04d}")
    excluded = result["domain"].eq("clipart") & indices.isin(EXCLUDED_CLIPART)
    return result.loc[~excluded].copy()


def load_extractor_module():
    name = "_feature_extraction_004"
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "004_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def jpeg85_batch(paths: list[Path]):
    import torch
    from PIL import Image

    arrays = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            buffer = BytesIO()
            rgb.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                arrays.append(
                    np.asarray(compressed.convert("RGB"), dtype=np.float32) / 255.0
                )
    batch = torch.from_numpy(np.stack(arrays))
    return batch.permute(0, 3, 1, 2).contiguous().mul_(2.0).sub_(1.0)


def calculate_variant(metadata, source_features, runtime, device) -> np.ndarray:
    import torch

    distances = np.empty(len(metadata), dtype=np.float32)
    row_ids = metadata["row_id"].to_numpy(dtype=np.int64)
    paths = [
        path if path.is_absolute() else ROOT / path
        for path in map(Path, metadata["image_path"].astype(str))
    ]
    with torch.inference_mode():
        for start in range(0, len(metadata), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(metadata))
            batch = jpeg85_batch(paths[start:end]).to(device, non_blocking=True)
            probe_features = runtime.extract(batch).float().cpu().numpy()
            processed_features = np.asarray(
                source_features[row_ids[start:end]], dtype=np.float32
            )
            difference = processed_features - probe_features
            distances[start:end] = np.sqrt(np.einsum(
                "ij,ij->i", difference, difference,
                dtype=np.float32, optimize=True,
            ))
    return distances


def main() -> None:
    metadata = pd.read_parquet(FEATURE_METADATA)
    for column in (
        "dataset_name", "domain", "data_index", "transformation",
        "transformation_factor",
    ):
        metadata[column] = metadata[column].astype(str)
    metadata = source_rows(
        metadata.loc[metadata["dataset_name"].eq("D_domainnet_clean")]
    )
    existing = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if OUTPUT_CSV.is_file():
        existing = source_rows(pd.read_csv(
            OUTPUT_CSV, usecols=OUTPUT_COLUMNS,
            dtype={name: str for name in OUTPUT_COLUMNS[:-1]},
        ))
    missing = []
    for operation, factor in VARIANTS:
        available = existing.loc[
            existing["benign_transformation"].eq(operation)
            & existing["benign_factor"].eq(factor)
        ]
        if len(available) != 2995 or available.duplicated(["domain", "data_index"]).any():
            missing.append((operation, factor))
    if not missing:
        print(f"Using existing benign TFI: {OUTPUT_CSV}")
        return

    extractor = load_extractor_module()
    device = extractor.DEVICE
    runtime = extractor.load_extractor("sd_vae")
    source_features = np.load(FEATURES_NPY, mmap_mode="r", allow_pickle=False)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    for operation, factor in missing:
        rows = metadata.loc[
            metadata["transformation"].eq(operation)
            & metadata["transformation_factor"].eq(factor)
        ].sort_values(["domain", "data_index"], kind="stable")
        if len(rows) != 2995 or rows.duplicated(["domain", "data_index"]).any():
            raise ValueError(f"Incomplete processed-clean input: {operation}/{factor}")
        calculated = rows[["domain", "data_index"]].copy()
        calculated["benign_transformation"] = operation
        calculated["benign_factor"] = factor
        calculated["latent_distance"] = calculate_variant(
            rows, source_features, runtime, device
        )
        existing = existing.loc[~(
            existing["benign_transformation"].eq(operation)
            & existing["benign_factor"].eq(factor)
        )]
        existing = pd.concat([existing, calculated], ignore_index=True)
        existing = existing.sort_values(
            OUTPUT_COLUMNS[:-1], kind="stable", ignore_index=True
        )
        existing.to_csv(OUTPUT_CSV, index=False, float_format="%.10g")
        print(f"{operation}/{factor}: {len(rows)} images")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
