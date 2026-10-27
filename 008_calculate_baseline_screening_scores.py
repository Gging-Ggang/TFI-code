from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from path_config import (
    CLEAN_DATASET,
    DATA_ROOT,
    GAUSSIAN_DATASET,
    GAUSSIAN_LPIPS_DATASET,
    NIGHTSHADE_DATASET,
    RESULT_ROOT,
)

try:
    import lpips
except ImportError as exc:
    raise ImportError("LPIPS is required. Install it with: pip install lpips") from exc


# =========================================================
# Hardcoded configuration
# =========================================================
TRANSFORMATION_ROOT = DATA_ROOT / "Transformation"
TFI_ROOT = RESULT_ROOT / "tfi"
OUTPUT_ROOT = RESULT_ROOT / "baseline"

SCORE_PATH = OUTPUT_ROOT / "008_baseline_scores.float32.npy"
ROW_PATH = OUTPUT_ROOT / "008_baseline_rows.csv"
COLUMN_PATH = OUTPUT_ROOT / "008_baseline_columns.csv"
MANIFEST_PATH = OUTPUT_ROOT / "008_baseline_manifest.json"

IMAGE_SIZE = 512
BATCH_SIZE = 512
LPIPS_NET = "alex"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
METRICS = ("pixel_rmse", "pixel_mae", "lpips", "frequency_rmse", "tfi")
IDENTITY_COLUMNS = ["cohort", "domain", "data_index"]
OVERWRITE = False
TFI_SOURCE = "005_tfi_sd_vae"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    cohort: str
    poisoning: str
    poisoning_factor: str


@dataclass(frozen=True)
class TransformationSpec:
    transformation: str
    transformation_factor: str
    directory: Path


DATASETS = (
    DatasetSpec(CLEAN_DATASET, "domainnet", "none", "none"),
    DatasetSpec(NIGHTSHADE_DATASET, "domainnet", "NS_linf", "0.04"),
    DatasetSpec(GAUSSIAN_DATASET, "domainnet", "GN_linf", "0.04"),
    DatasetSpec(
        GAUSSIAN_LPIPS_DATASET,
        "domainnet",
        "GN_star_linf",
        "0.04",
    ),
)

TRANSFORMATIONS = (
    *(
        TransformationSpec("downscaling", str(factor), Path("downscaling") / f"factor_{factor}")
        for factor in ("0.75", "0.6", "0.5", "0.4", "0.3", "0.25")
    ),
    *(
        TransformationSpec(
            "gaussian_blur",
            sigma,
            Path("gaussian blur") / f"sigma_{float(sigma):.1f}",
        )
        for sigma in ("0.5", "1.0", "2.0", "3.0")
    ),
    *(
        TransformationSpec("jpeg_comp", quality, Path("jpeg_comp") / f"quality_{quality}")
        for quality in ("95", "85", "75", "50")
    ),
)


def infer_domain(path: Path, dataset: DatasetSpec) -> str:
    if dataset.cohort == "domainnet":
        return path.parent.name
    raise ValueError(f"Unknown cohort: {dataset.cohort}")


def infer_data_index(path: Path) -> str:
    suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Could not infer numeric data index: {path}")
    return f"{int(suffix):04d}"


def discover_images(root: Path, dataset: DatasetSpec) -> dict[tuple[str, str], Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {root}")
    images: dict[tuple[str, str], Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = (infer_domain(path, dataset), infer_data_index(path))
        if key in images:
            raise ValueError(f"Duplicate image identity {key}: {images[key]} and {path}")
        images[key] = path
    if not images:
        raise FileNotFoundError(f"No images found in: {root}")
    return images


def validate_and_collect_paths(
    dataset: DatasetSpec,
    transformation: TransformationSpec,
) -> tuple[list[tuple[str, str]], list[Path], list[Path]]:
    originals = discover_images(DATA_ROOT / dataset.dataset_name, dataset)
    transformed = discover_images(
        TRANSFORMATION_ROOT / transformation.directory / dataset.dataset_name,
        dataset,
    )
    if originals.keys() != transformed.keys():
        missing = sorted(originals.keys() - transformed.keys())
        extra = sorted(transformed.keys() - originals.keys())
        raise ValueError(
            f"Pair mismatch for {dataset.dataset_name}/{transformation.transformation}/"
            f"{transformation.transformation_factor}: "
            f"{len(missing)} missing transformed and {len(extra)} extra transformed; "
            f"examples missing={missing[:5]}, extra={extra[:5]}"
        )
    keys = sorted(originals)
    return keys, [originals[key] for key in keys], [transformed[key] for key in keys]


def build_rows() -> pd.DataFrame:
    identities_by_cohort: dict[str, set[tuple[str, str]]] = {}
    reference_dataset_by_cohort: dict[str, str] = {}
    for dataset in DATASETS:
        images = discover_images(DATA_ROOT / dataset.dataset_name, dataset)
        identities = set(images)
        if dataset.cohort not in identities_by_cohort:
            identities_by_cohort[dataset.cohort] = identities
            reference_dataset_by_cohort[dataset.cohort] = dataset.dataset_name
            continue

        reference = identities_by_cohort[dataset.cohort]
        if identities != reference:
            missing = sorted(reference - identities)
            extra = sorted(identities - reference)
            raise ValueError(
                f"Identity mismatch within cohort '{dataset.cohort}': "
                f"{dataset.dataset_name} differs from "
                f"{reference_dataset_by_cohort[dataset.cohort]}; "
                f"{len(missing)} missing and {len(extra)} extra identities; "
                f"examples missing={missing[:5]}, extra={extra[:5]}"
            )

    rows = {
        (cohort, domain, data_index)
        for cohort, identities in identities_by_cohort.items()
        for domain, data_index in identities
    }
    frame = pd.DataFrame(sorted(rows), columns=IDENTITY_COLUMNS)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def build_columns() -> pd.DataFrame:
    records: list[dict] = []
    column_id = 0
    for dataset in DATASETS:
        for transformation in TRANSFORMATIONS:
            for metric in METRICS:
                records.append(
                    {
                        "column_id": column_id,
                        "dataset_name": dataset.dataset_name,
                        "cohort": dataset.cohort,
                        "poisoning": dataset.poisoning,
                        "poisoning_factor": dataset.poisoning_factor,
                        "transformation": transformation.transformation,
                        "transformation_factor": transformation.transformation_factor,
                        "metric": metric,
                    }
                )
                column_id += 1
    return pd.DataFrame(records)


def atomic_save_npy(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, array, allow_pickle=False)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_rgb_batch(paths: list[Path]) -> torch.Tensor:
    arrays: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, found "
                    f"{rgb.width}x{rgb.height}: {path}"
                )
            # Copy makes the NumPy storage writable for torch.from_numpy.
            arrays.append(np.asarray(rgb, dtype=np.float32).copy() / 255.0)
    batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    if DEVICE.type == "cuda":
        batch = batch.pin_memory()
    return batch.to(DEVICE, non_blocking=True)


def build_radial_bins() -> tuple[torch.Tensor, torch.Tensor, int]:
    coordinates = torch.arange(IMAGE_SIZE, device=DEVICE) - IMAGE_SIZE // 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    bins = torch.round(torch.sqrt(yy.float().square() + xx.float().square()))
    bins = bins.to(torch.long).reshape(-1)
    maximum = int(bins.max().item())
    counts = torch.bincount(bins, minlength=maximum + 1).to(torch.float32)
    return bins, counts, maximum


@torch.inference_mode()
def radial_log_power_profile(
    batch: torch.Tensor,
    radial_bins: torch.Tensor,
    radial_counts: torch.Tensor,
    maximum_bin: int,
) -> torch.Tensor:
    # [B, 3, H, W] -> channel-averaged radial log-power profile [B, K].
    spectrum = torch.fft.fftshift(torch.fft.fft2(batch), dim=(-2, -1))
    log_power = torch.log1p(spectrum.abs().square())
    flat = log_power.reshape(batch.shape[0] * 3, -1)
    expanded_bins = radial_bins.unsqueeze(0).expand(flat.shape[0], -1)
    sums = torch.zeros(
        (flat.shape[0], maximum_bin + 1),
        device=batch.device,
        dtype=flat.dtype,
    )
    sums.scatter_add_(1, expanded_bins, flat)
    profiles = sums / radial_counts.clamp_min(1).unsqueeze(0)
    profiles = profiles.reshape(batch.shape[0], 3, maximum_bin + 1).mean(dim=1)
    return profiles[:, 1:]  # Exclude DC (b=0).


def load_tfi_sources() -> tuple[pd.DataFrame, np.ndarray]:
    metadata_path = TFI_ROOT / "tfi.csv"
    metric_path = TFI_ROOT / "tfi.float32.npy"
    if not metadata_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError(
            "TFI reuses the SD VAE latent distances produced by "
            "005_calculate_tfi.py. "
            f"Missing: {metadata_path if not metadata_path.is_file() else metric_path}"
        )
    required = [
        "domain",
        "data_index",
        "feature_extractor",
        "poisoning",
        "poisoning_factor",
    ]
    metadata = pd.read_csv(
        metadata_path,
        usecols=required,
        dtype={column: "string" for column in required},
    )
    metrics = np.load(metric_path, mmap_mode="r", allow_pickle=False)
    expected_metric_columns = len(TRANSFORMATIONS) * 2
    if metrics.shape != (len(metadata), expected_metric_columns):
        raise ValueError(
            "005 TFI metadata/array mismatch: "
            f"{metrics.shape} != {(len(metadata), expected_metric_columns)}"
        )
    metadata.insert(0, "metric_row", np.arange(len(metadata), dtype=np.int64))
    metadata = metadata.loc[
        metadata["feature_extractor"].eq("sd_vae")
    ].copy()
    metadata["data_index"] = metadata["data_index"].map(
        lambda value: f"{int(value):04d}"
    )
    duplicated = metadata.duplicated(
        ["poisoning", "poisoning_factor", "domain", "data_index"],
        keep=False,
    )
    if duplicated.any():
        examples = metadata.loc[
            duplicated,
            ["poisoning", "poisoning_factor", "domain", "data_index"],
        ].head(10)
        raise ValueError(
            "Duplicate SD VAE rows in 005 TFI output:\n"
            + examples.to_string(index=False)
        )
    return metadata, metrics


def tfi_scores(
    metadata: pd.DataFrame,
    metrics: np.ndarray,
    dataset: DatasetSpec,
    transformation: TransformationSpec,
    keys: list[tuple[str, str]],
) -> np.ndarray:
    subset = metadata.loc[
        metadata["poisoning"].eq(dataset.poisoning)
        & metadata["poisoning_factor"].eq(dataset.poisoning_factor),
        ["metric_row", "domain", "data_index"],
    ].copy()
    key_index = pd.MultiIndex.from_tuples(keys, names=["domain", "data_index"])
    subset = subset.set_index(["domain", "data_index"])
    missing = key_index.difference(subset.index)
    if len(missing):
        raise ValueError(
            f"Missing {len(missing)} existing 005 TFI rows for "
            f"{dataset.dataset_name}: {list(missing[:5])}"
        )
    metric_rows = subset.loc[key_index, "metric_row"].to_numpy(dtype=np.int64)
    transformation_index = TRANSFORMATIONS.index(transformation)
    return np.asarray(
        metrics[metric_rows, transformation_index],
        dtype=np.float32,
    )


@torch.inference_mode()
def image_scores(
    original_paths: list[Path],
    transformed_paths: list[Path],
    lpips_model: torch.nn.Module,
    radial_bins: torch.Tensor,
    radial_counts: torch.Tensor,
    maximum_bin: int,
) -> np.ndarray:
    result = np.empty((len(original_paths), 4), dtype=np.float32)
    for start in range(0, len(original_paths), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(original_paths))
        original = load_rgb_batch(original_paths[start:end])
        transformed = load_rgb_batch(transformed_paths[start:end])
        difference = original - transformed

        result[start:end, 0] = (
            difference.square().mean(dim=(1, 2, 3)).sqrt().cpu().numpy()
        )
        result[start:end, 1] = difference.abs().mean(dim=(1, 2, 3)).cpu().numpy()
        result[start:end, 2] = (
            lpips_model(original.mul(2).sub(1), transformed.mul(2).sub(1))
            .reshape(end - start, -1)
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        original_profile = radial_log_power_profile(
            original, radial_bins, radial_counts, maximum_bin
        )
        transformed_profile = radial_log_power_profile(
            transformed, radial_bins, radial_counts, maximum_bin
        )
        result[start:end, 3] = (
            (original_profile - transformed_profile)
            .square()
            .mean(dim=1)
            .sqrt()
            .cpu()
            .numpy()
        )
    return result


def column_key(row) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.dataset_name),
        str(row.cohort),
        str(row.poisoning),
        str(row.poisoning_factor),
        str(row.transformation),
        str(row.transformation_factor),
        str(row.metric),
    )


def load_or_initialize_scores(
    rows: pd.DataFrame,
    columns: pd.DataFrame,
) -> np.ndarray:
    output_paths = (SCORE_PATH, ROW_PATH, COLUMN_PATH)
    existing = [path.is_file() for path in output_paths]
    if OVERWRITE or not any(existing):
        scores = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
        rows.to_csv(ROW_PATH, index=False)
        columns.to_csv(COLUMN_PATH, index=False)
        atomic_save_npy(scores, SCORE_PATH)
        return scores
    if not all(existing):
        missing = [str(path) for path, exists in zip(output_paths, existing) if not exists]
        raise FileNotFoundError(
            "Incomplete existing 008 output set; refusing an unsafe migration. "
            f"Missing: {missing}"
        )

    old_rows = pd.read_csv(
        ROW_PATH,
        dtype={"cohort": "string", "domain": "string", "data_index": "string"},
    )
    old_columns = pd.read_csv(
        COLUMN_PATH,
        dtype={
            "dataset_name": "string",
            "cohort": "string",
            "poisoning": "string",
            "poisoning_factor": "string",
            "transformation": "string",
            "transformation_factor": "string",
            "metric": "string",
        },
    )
    old_scores = np.load(SCORE_PATH, mmap_mode="r", allow_pickle=False)
    if old_scores.shape != (len(old_rows), len(old_columns)):
        raise ValueError(
            "Existing 008 score/metadata shape mismatch: "
            f"{old_scores.shape} != {(len(old_rows), len(old_columns))}"
        )
    if not np.array_equal(
        old_rows["row_id"].to_numpy(dtype=np.int64),
        np.arange(len(old_rows), dtype=np.int64),
    ):
        raise ValueError("Existing 008 row_id is not sequential")
    if not np.array_equal(
        old_columns["column_id"].to_numpy(dtype=np.int64),
        np.arange(len(old_columns), dtype=np.int64),
    ):
        raise ValueError("Existing 008 column_id is not sequential")

    identity_columns = ["cohort", "domain", "data_index"]
    old_identities = old_rows.loc[:, identity_columns].astype(str)
    new_identities = rows.loc[:, identity_columns].astype(str)
    if not old_identities.equals(new_identities):
        raise ValueError(
            "Existing 008 row identities differ from the current experiment. "
            "Automatic score migration is unsafe."
        )

    new_column_lookup = {
        column_key(row): int(row.column_id)
        for row in columns.itertuples(index=False)
    }
    migrated = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    retained = 0
    dropped = 0
    for old_column in old_columns.itertuples(index=False):
        destination = new_column_lookup.get(column_key(old_column))
        if destination is None:
            dropped += 1
            continue
        migrated[:, destination] = old_scores[:, int(old_column.column_id)]
        retained += 1

    rows.to_csv(ROW_PATH, index=False)
    columns.to_csv(COLUMN_PATH, index=False)
    atomic_save_npy(migrated, SCORE_PATH)
    print(
        f"Migrated existing scores: {retained} columns retained, "
        f"{len(columns) - retained} new, {dropped} obsolete columns removed"
    )
    return migrated


def validate_finished(
    scores: np.ndarray,
    rows: pd.DataFrame,
    columns: pd.DataFrame,
) -> None:
    expected_shape = (len(rows), len(columns))
    if scores.shape != expected_shape:
        raise ValueError(f"Score shape mismatch: {scores.shape} != {expected_shape}")
    applicable = np.zeros(expected_shape, dtype=bool)
    for dataset in DATASETS:
        row_mask = rows["cohort"].eq(dataset.cohort).to_numpy()
        column_mask = columns["dataset_name"].eq(dataset.dataset_name).to_numpy()
        applicable[np.ix_(row_mask, column_mask)] = True
    if not np.isfinite(scores[applicable]).all():
        raise ValueError("Applicable score cells contain non-finite values")
    if not np.isnan(scores[~applicable]).all():
        raise ValueError("Non-applicable score cells must be NaN")


def main() -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU is required for LPIPS and FFT score extraction.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    columns = build_columns()
    scores = load_or_initialize_scores(rows, columns)
    row_lookup = {
        (row.cohort, row.domain, row.data_index): int(row.row_id)
        for row in rows.itertuples(index=False)
    }

    pending_jobs: list[
        tuple[DatasetSpec, TransformationSpec, np.ndarray, np.ndarray]
    ] = []
    for dataset in DATASETS:
        dataset_row_ids = rows.loc[
            rows["cohort"].eq(dataset.cohort),
            "row_id",
        ].to_numpy(dtype=np.int64)
        for transformation in TRANSFORMATIONS:
            column_mask = (
                columns["dataset_name"].eq(dataset.dataset_name)
                & columns["transformation"].eq(transformation.transformation)
                & columns["transformation_factor"].eq(
                    transformation.transformation_factor
                )
            )
            column_ids = columns.loc[
                column_mask,
                "column_id",
            ].to_numpy(dtype=np.int64)
            if len(column_ids) != len(METRICS):
                raise ValueError("Internal column mapping error")
            if np.isfinite(scores[np.ix_(dataset_row_ids, column_ids)]).all():
                print(
                    f"[SKIP] {dataset.dataset_name} | "
                    f"{transformation.transformation} "
                    f"{transformation.transformation_factor}"
                )
                continue
            pending_jobs.append(
                (dataset, transformation, dataset_row_ids, column_ids)
            )

    if not pending_jobs:
        validate_finished(scores, rows, columns)
        print(f"All {len(DATASETS) * len(TRANSFORMATIONS)} jobs already complete")
        return

    lpips_model = lpips.LPIPS(net=LPIPS_NET).to(DEVICE).eval()
    radial_bins, radial_counts, maximum_bin = build_radial_bins()
    latent_metadata, latent_metrics = load_tfi_sources()

    for job, (
        dataset,
        transformation,
        expected_row_ids,
        column_ids,
    ) in enumerate(pending_jobs, start=1):
        keys, original_paths, transformed_paths = validate_and_collect_paths(
            dataset,
            transformation,
        )
        row_ids = np.asarray(
            [
                row_lookup[(dataset.cohort, domain, data_index)]
                for domain, data_index in keys
            ],
            dtype=np.int64,
        )
        if not np.array_equal(np.sort(row_ids), np.sort(expected_row_ids)):
            raise ValueError(
                f"Unexpected row identities for {dataset.dataset_name}"
            )
        image_values = image_scores(
            original_paths,
            transformed_paths,
            lpips_model,
            radial_bins,
            radial_counts,
            maximum_bin,
        )
        tfi = tfi_scores(
            latent_metadata,
            latent_metrics,
            dataset,
            transformation,
            keys,
        )
        values = np.column_stack([image_values, tfi]).astype(
            np.float32,
            copy=False,
        )
        scores[np.ix_(row_ids, column_ids)] = values
        atomic_save_npy(scores, SCORE_PATH)
        print(
            f"[CALCULATE {job}/{len(pending_jobs)}] {dataset.dataset_name} | "
            f"{transformation.transformation} "
            f"{transformation.transformation_factor} | {len(keys)} images | "
            "checkpoint saved"
        )

    validate_finished(scores, rows, columns)
    manifest = {
        "score_path": SCORE_PATH.name,
        "row_path": ROW_PATH.name,
        "column_path": COLUMN_PATH.name,
        "shape": [len(rows), len(columns)],
        "dtype": "float32",
        "lpips_backbone": LPIPS_NET,
        "image_size": IMAGE_SIZE,
        "frequency_bins_excluding_dc": maximum_bin,
        "metrics": list(METRICS),
        "tfi_source": TFI_SOURCE,
        "non_applicable_value": "NaN",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved scores: {SCORE_PATH} {scores.shape}")
    print(f"Saved row metadata: {ROW_PATH}")
    print(f"Saved column metadata: {COLUMN_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
