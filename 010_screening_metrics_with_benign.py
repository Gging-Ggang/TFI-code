from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from path_config import CLEAN_DATASET

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as exc:
    raise ImportError("Install scikit-learn with: pip install scikit-learn") from exc


# =========================================================
# Hardcoded configuration
# =========================================================
ROOT = Path(__file__).resolve().parent
TFI_ROOT = ROOT / "result" / "tfi"
SCREENING_ROOT = ROOT / "result" / "screening"

BASE_METADATA_CSV = TFI_ROOT / "tfi.csv"
BASE_VALUES_NPY = TFI_ROOT / "tfi.float32.npy"
FEATURE_METADATA_PARQUET = ROOT / "result" / "features" / "feature_metadata.parquet"
SD_VAE_FEATURES_NPY = (
    ROOT / "result" / "features" / "sd_vae" / "features.float16.npy"
)

# One row per (domain, data_index, benign_transformation, benign_factor).
# TFI must be:
# ||f(P(x)) - f(JPEG_Q85(P(x)))||_2 measured with the SD VAE.
BENIGN_TFI_CSV = TFI_ROOT / "010_benign_tfi_sd_vae_jpeg85.csv"
BENIGN_TFI_VALIDATION_JSON = (
    TFI_ROOT / "010_benign_tfi_sd_vae_jpeg85.validation.json"
)
AUTO_CALCULATE_BENIGN_TFI = True
BENIGN_TFI_BATCH_SIZE = 512

DOMAIN_CSV = SCREENING_ROOT / "010_screening_benign_domain_stratified.csv"
POOLED_CSV = SCREENING_ROOT / "010_screening_benign_pooled_random.csv"
DOMAIN_SUMMARY_CSV = (
    SCREENING_ROOT / "010_screening_benign_summary_domain_stratified.csv"
)
POOLED_SUMMARY_CSV = (
    SCREENING_ROOT / "010_screening_benign_summary_pooled_random.csv"
)

OVERWRITE_RAW = True
RATIOS = (0.10,)
SEED_RATIO_ORDER = (0.01, 0.05, 0.10, 0.20)
BENIGN_RATIO: float | None = 0.20
BUDGETS = (0.01, 0.05, 0.10, 0.20)
REPEATS = 50
SEED = 0

FEATURE_EXTRACTOR = "sd_vae"
PROBE_TRANSFORMATION = "jpeg_comp"
PROBE_FACTOR = "85"
PROBE_COLUMN = "ld_jpeg_comp_85"
DOMAINNET_DOMAINS = (
    "real",
    "painting",
    "clipart",
    "infograph",
    "sketch",
    "quickdraw",
)
EXPECTED_DOMAIN_COUNTS = {
    domain: 10
    for domain in DOMAINNET_DOMAINS
}
EXPECTED_POOL_SIZE = sum(EXPECTED_DOMAIN_COUNTS.values())

EXTRACT_FEATURES_CODE = ROOT / "004_feature_extraction.py"
MODEL_CHECK_MAX_RELATIVE_L2 = 1e-3
MODEL_CHECK_MAX_ABSOLUTE_ERROR = 5e-2

TRANSFORMATIONS = (
    ("downscaling", "0.75"),
    ("downscaling", "0.6"),
    ("downscaling", "0.5"),
    ("downscaling", "0.4"),
    ("downscaling", "0.3"),
    ("downscaling", "0.25"),
    ("gaussian_blur", "0.5"),
    ("gaussian_blur", "1.0"),
    ("gaussian_blur", "2.0"),
    ("gaussian_blur", "3.0"),
    ("jpeg_comp", "95"),
    ("jpeg_comp", "85"),
    ("jpeg_comp", "75"),
    ("jpeg_comp", "50"),
)
LD_COLUMNS = [
    f"ld_{transformation}_{factor}"
    for transformation, factor in TRANSFORMATIONS
]
ANGLE_COLUMNS = [
    column.replace("ld_", "angle_", 1)
    for column in LD_COLUMNS
]
METRIC_COLUMNS = [*LD_COLUMNS, *ANGLE_COLUMNS]
PROBE_COLUMN_INDEX = METRIC_COLUMNS.index(PROBE_COLUMN)

NIGHTSHADE_ATTACKS = (("NS_linf", "0.04"),)

# Names and factors must match BENIGN_TFI_CSV exactly.
BENIGN_VARIANTS = TRANSFORMATIONS
EXPECTED_BENIGN_ROWS = EXPECTED_POOL_SIZE * len(BENIGN_VARIANTS)

BASE_COLUMNS = [
    "domain",
    "data_index",
    "feature_extractor",
    "poisoning",
    "poisoning_factor",
]
BENIGN_COLUMNS = [
    "domain",
    "data_index",
    "benign_transformation",
    "benign_factor",
    "latent_distance",
]


def exclude_identities(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


def validate_domainnet_pool(data: pd.DataFrame, name: str) -> None:
    counts = data.groupby("domain", observed=True).size()
    expected = pd.Series(
        EXPECTED_DOMAIN_COUNTS,
        dtype=np.int64,
    )
    expected.index = pd.Index(expected.index, dtype="string", name="domain")
    counts = counts.reindex(expected.index, fill_value=0)
    if len(data) != EXPECTED_POOL_SIZE or not counts.eq(expected).all():
        raise ValueError(
            f"{name} must contain exactly {EXPECTED_POOL_SIZE} DomainNet rows "
            f"with expected per-domain counts; found {len(data)} rows:\n"
            + counts.to_string()
        )


def load_base_scores() -> pd.DataFrame:
    metadata = pd.read_csv(
        BASE_METADATA_CSV,
        usecols=BASE_COLUMNS,
        dtype={column: "string" for column in BASE_COLUMNS},
    )
    values = np.load(BASE_VALUES_NPY, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(metadata), len(METRIC_COLUMNS)):
        raise ValueError(
            f"Base metric shape mismatch: expected "
            f"{(len(metadata), len(METRIC_COLUMNS))}, found {values.shape}"
        )

    selected = metadata.loc[
        metadata["feature_extractor"].eq(FEATURE_EXTRACTOR)
        & metadata["domain"].isin(DOMAINNET_DOMAINS)
    ].copy()
    selected["latent_distance"] = np.asarray(
        values[selected.index.to_numpy(), PROBE_COLUMN_INDEX],
        dtype=np.float32,
    )
    return exclude_identities(selected).reset_index(drop=True)


def atomic_save_csv(data: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{destination.stem}.",
            dir=destination.parent,
            delete=False,
            encoding="utf-8",
        ) as temporary:
            temporary_path = Path(temporary.name)
            data.to_csv(temporary, index=False, float_format="%.10g")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_004_feature_extraction_module() -> ModuleType:
    module_name = "_feature_extraction_004"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    specification = importlib.util.spec_from_file_location(
        module_name, EXTRACT_FEATURES_CODE
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import feature extractor: {EXTRACT_FEATURES_CODE}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def load_benign_feature_metadata() -> pd.DataFrame:
    required = [
        "row_id",
        "image_path",
        "dataset_name",
        "domain",
        "data_index",
        "transformation",
        "transformation_factor",
    ]
    metadata = pd.read_parquet(
        FEATURE_METADATA_PARQUET,
        columns=required,
        engine="pyarrow",
    )
    string_columns = [column for column in required if column != "row_id"]
    for column in string_columns:
        metadata[column] = metadata[column].astype("string")

    variant_index = pd.MultiIndex.from_tuples(
        BENIGN_VARIANTS,
        names=["transformation", "transformation_factor"],
    )
    row_variant_index = pd.MultiIndex.from_frame(
        metadata[["transformation", "transformation_factor"]]
    )
    selected = metadata.loc[
        metadata["dataset_name"].eq(CLEAN_DATASET)
        & metadata["domain"].isin(DOMAINNET_DOMAINS)
        & row_variant_index.isin(variant_index)
    ].copy()
    selected["data_index"] = selected["data_index"].map(
        lambda value: f"{int(value):04d}"
    )
    selected = exclude_identities(selected)

    duplicated = selected.duplicated(
        ["domain", "data_index", "transformation", "transformation_factor"],
        keep=False,
    )
    if duplicated.any():
        examples = selected.loc[
            duplicated,
            ["domain", "data_index", "transformation", "transformation_factor"],
        ].head(10)
        raise ValueError(
            "Duplicate benign feature metadata rows:\n"
            + examples.to_string(index=False)
        )

    for transformation, factor in BENIGN_VARIANTS:
        variant = selected.loc[
            selected["transformation"].eq(transformation)
            & selected["transformation_factor"].eq(factor)
        ]
        validate_domainnet_pool(
            variant, f"Feature metadata for benign {transformation}/{factor}"
        )
    if len(selected) != EXPECTED_BENIGN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_BENIGN_ROWS} benign feature rows, "
            f"found {len(selected)}"
        )
    return selected


def existing_benign_tfi() -> pd.DataFrame:
    if (
        not BENIGN_TFI_CSV.is_file()
        or not BENIGN_TFI_VALIDATION_JSON.is_file()
    ):
        return pd.DataFrame(columns=BENIGN_COLUMNS)
    validation = json.loads(
        BENIGN_TFI_VALIDATION_JSON.read_text(encoding="utf-8")
    )
    feature_stat = SD_VAE_FEATURES_NPY.stat()
    if (
        validation.get("source_feature_size") != feature_stat.st_size
        or validation.get("source_feature_mtime_ns") != feature_stat.st_mtime_ns
    ):
        return pd.DataFrame(columns=BENIGN_COLUMNS)
    data = pd.read_csv(
        BENIGN_TFI_CSV,
        usecols=BENIGN_COLUMNS,
        dtype={
            "domain": "string",
            "data_index": "string",
            "benign_transformation": "string",
            "benign_factor": "string",
        },
    )
    data["latent_distance"] = pd.to_numeric(
        data["latent_distance"], errors="raise"
    )
    duplicated = data.duplicated(
        [
            "domain",
            "data_index",
            "benign_transformation",
            "benign_factor",
        ],
        keep=False,
    )
    if duplicated.any():
        raise ValueError("Existing benign TFI CSV contains duplicate rows")
    return exclude_identities(data).reset_index(drop=True)


def resolved_image_paths(metadata: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for value in metadata["image_path"].astype(str):
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing benign input image: {path}")
        paths.append(path)
    return paths


def jpeg85_probe_batch(image_paths: list[Path]):
    import torch
    from PIL import Image

    arrays: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if rgb.size != (512, 512):
                raise ValueError(
                    f"Expected 512x512 benign input, found {rgb.size}: {image_path}"
                )
            buffer = BytesIO()
            rgb.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                arrays.append(
                    np.asarray(
                        compressed.convert("RGB"),
                        dtype=np.float32,
                    )
                    / 255.0
                )
    batch = torch.from_numpy(np.stack(arrays))
    return batch.permute(0, 3, 1, 2).contiguous().mul_(2.0).sub_(1.0)


def calculate_benign_variant_tfi(
    metadata: pd.DataFrame,
    source_features: np.ndarray,
    extractor_runtime,
    device,
) -> np.ndarray:
    import torch

    paths = resolved_image_paths(metadata)
    row_ids = metadata["row_id"].to_numpy(dtype=np.int64)
    distances = np.empty(len(metadata), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(metadata), BENIGN_TFI_BATCH_SIZE):
            end = min(start + BENIGN_TFI_BATCH_SIZE, len(metadata))
            probe_batch = jpeg85_probe_batch(paths[start:end]).to(
                device, non_blocking=True
            )
            probe_features = (
                extractor_runtime.extract(probe_batch)
                .float()
                .cpu()
                .numpy()
            )
            benign_features = np.asarray(
                source_features[row_ids[start:end]], dtype=np.float32
            )
            difference = benign_features - probe_features
            squared_distance = np.einsum(
                "ij,ij->i",
                difference,
                difference,
                dtype=np.float32,
                optimize=True,
            )
            distances[start:end] = np.sqrt(squared_distance)
    return distances


def verify_004_model_matches_stored_features(
    feature_metadata: pd.DataFrame,
    source_features: np.ndarray,
    extractor_runtime,
    device,
) -> tuple[int, float, float]:
    import torch

    reference = feature_metadata.sort_values(
        [
            "transformation",
            "transformation_factor",
            "domain",
            "data_index",
        ],
        kind="mergesort",
    ).iloc[0]
    reference_path = resolved_image_paths(reference.to_frame().T)[0]
    row_id = int(reference["row_id"])

    with torch.inference_mode():
        batch = extractor_runtime.preprocess([reference_path]).to(
            device, non_blocking=True
        )
        recalculated = (
            extractor_runtime.extract(batch)
            .flatten()
            .float()
            .cpu()
            .numpy()
        )
    stored = np.asarray(source_features[row_id], dtype=np.float32)
    if recalculated.shape != stored.shape:
        raise ValueError(
            f"004 model-check feature shape mismatch: "
            f"{recalculated.shape} != {stored.shape}"
        )

    difference = recalculated - stored
    l2_difference = float(np.linalg.norm(difference))
    stored_norm = float(np.linalg.norm(stored))
    relative_l2 = l2_difference / max(stored_norm, np.finfo(np.float32).eps)
    maximum_error = float(np.max(np.abs(difference)))
    if (
        relative_l2 > MODEL_CHECK_MAX_RELATIVE_L2
        or maximum_error > MODEL_CHECK_MAX_ABSOLUTE_ERROR
    ):
        raise RuntimeError(
            "The SD VAE currently selected by 004_feature_extraction.py does not "
            "match the model used to create the stored SD VAE features. "
            "Refusing to mix incompatible feature spaces.\n"
            f"Check row_id={row_id}, image={reference_path}\n"
            f"L2 difference={l2_difference:.6f}, "
            f"relative L2={relative_l2:.6g}, "
            f"maximum absolute error={maximum_error:.6f}\n"
            "Regenerate the stored SD VAE features with the model pinned in "
            "004_feature_extraction.py before recalculating benign TFI."
        )
    print(
        "Verified 004 SD VAE against stored features: "
        f"relative L2={relative_l2:.6g}, "
        f"max abs={maximum_error:.6f}"
    )
    return row_id, relative_l2, maximum_error


def save_benign_tfi_validation(
    row_id: int,
    relative_l2: float,
    maximum_error: float,
) -> None:
    feature_stat = SD_VAE_FEATURES_NPY.stat()
    validation = {
        "model_loader": EXTRACT_FEATURES_CODE.name,
        "verified_row_id": row_id,
        "relative_l2": relative_l2,
        "maximum_absolute_error": maximum_error,
        "source_feature_size": feature_stat.st_size,
        "source_feature_mtime_ns": feature_stat.st_mtime_ns,
    }
    BENIGN_TFI_VALIDATION_JSON.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_benign_tfi() -> None:
    existing = existing_benign_tfi()
    feature_metadata = load_benign_feature_metadata()
    identity = ["domain", "data_index"]
    missing_variants: list[tuple[str, str]] = []

    for transformation, factor in BENIGN_VARIANTS:
        expected = feature_metadata.loc[
            feature_metadata["transformation"].eq(transformation)
            & feature_metadata["transformation_factor"].eq(factor),
            identity,
        ]
        available = existing.loc[
            existing["benign_transformation"].eq(transformation)
            & existing["benign_factor"].eq(factor),
            identity,
        ]
        expected_index = pd.MultiIndex.from_frame(expected)
        available_index = pd.MultiIndex.from_frame(available)
        missing = expected_index.difference(available_index)
        extra = available_index.difference(expected_index)
        if (
            len(available) != EXPECTED_POOL_SIZE
            or len(missing)
            or len(extra)
        ):
            missing_variants.append((transformation, factor))

    if not missing_variants:
        print(f"Using existing benign TFI: {BENIGN_TFI_CSV}")
        return
    if not AUTO_CALCULATE_BENIGN_TFI:
        raise FileNotFoundError(
            f"Missing or incomplete benign TFI table: {BENIGN_TFI_CSV}"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required to calculate missing benign TFI scores"
        )
    extractor_004 = load_004_feature_extraction_module()
    device = torch.device(extractor_004.DEVICE)
    if device.type != "cuda":
        raise RuntimeError(
            "004_feature_extraction.py did not select a CUDA device"
        )
    source_features = np.load(
        SD_VAE_FEATURES_NPY, mmap_mode="r", allow_pickle=False
    )
    if source_features.ndim != 2 or source_features.shape[0] <= int(
        feature_metadata["row_id"].max()
    ):
        raise ValueError(
            f"SD VAE feature array does not cover benign rows: "
            f"{source_features.shape}"
        )

    extractor_runtime = extractor_004.load_extractor("sd_vae")
    validation_result = verify_004_model_matches_stored_features(
        feature_metadata,
        source_features,
        extractor_runtime,
        device,
    )
    for job, (transformation, factor) in enumerate(
        missing_variants, start=1
    ):
        variant = feature_metadata.loc[
            feature_metadata["transformation"].eq(transformation)
            & feature_metadata["transformation_factor"].eq(factor)
        ].sort_values(["domain", "data_index"], kind="mergesort")
        print(
            f"[BENIGN TFI {job}/{len(missing_variants)}] "
            f"{transformation} {factor} | {len(variant)} images"
        )
        distances = calculate_benign_variant_tfi(
            variant, source_features, extractor_runtime, device
        )
        calculated = pd.DataFrame(
            {
                "domain": variant["domain"].to_numpy(),
                "data_index": variant["data_index"].to_numpy(),
                "benign_transformation": transformation,
                "benign_factor": factor,
                "latent_distance": distances,
            }
        )
        existing = existing.loc[
            ~(
                existing["benign_transformation"].eq(transformation)
                & existing["benign_factor"].eq(factor)
            )
        ]
        existing = pd.concat([existing, calculated], ignore_index=True)
        existing = existing.sort_values(
            [
                "benign_transformation",
                "benign_factor",
                "domain",
                "data_index",
            ],
            kind="mergesort",
            ignore_index=True,
        )
        atomic_save_csv(existing, BENIGN_TFI_CSV)
        save_benign_tfi_validation(*validation_result)
        print(f"Checkpoint saved: {BENIGN_TFI_CSV}")

    del extractor_runtime
    torch.cuda.empty_cache()


def load_benign_scores() -> pd.DataFrame:
    ensure_benign_tfi()

    benign = pd.read_csv(
        BENIGN_TFI_CSV,
        usecols=BENIGN_COLUMNS,
        dtype={
            "domain": "string",
            "data_index": "string",
            "benign_transformation": "string",
            "benign_factor": "string",
        },
    )
    benign["latent_distance"] = pd.to_numeric(
        benign["latent_distance"], errors="raise"
    )
    benign = benign.loc[
        benign["domain"].isin(DOMAINNET_DOMAINS)
    ].copy()
    benign = exclude_identities(benign)
    if not np.isfinite(benign["latent_distance"].to_numpy()).all():
        raise ValueError("Benign TFI table contains non-finite latent distances")

    key = [
        "domain",
        "data_index",
        "benign_transformation",
        "benign_factor",
    ]
    duplicated = benign.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate benign TFI rows:\n"
            + benign.loc[duplicated, key].head(10).to_string(index=False)
        )
    return benign


def paired_scores(
    base: pd.DataFrame,
    benign: pd.DataFrame,
    attack: tuple[str, str],
    benign_variant: tuple[str, str],
) -> pd.DataFrame:
    poisoning, poisoning_factor = attack
    benign_transformation, benign_factor = benign_variant
    identity = ["domain", "data_index"]
    base = exclude_identities(base)
    benign = exclude_identities(benign)

    clean = base.loc[
        base["poisoning"].eq("none") & base["poisoning_factor"].eq("none"),
        [*identity, "latent_distance"],
    ].rename(columns={"latent_distance": "clean_score"})
    poison = base.loc[
        base["poisoning"].eq(poisoning)
        & base["poisoning_factor"].eq(poisoning_factor),
        [*identity, "latent_distance"],
    ].rename(columns={"latent_distance": "poison_score"})
    processed = benign.loc[
        benign["benign_transformation"].eq(benign_transformation)
        & benign["benign_factor"].eq(benign_factor),
        [*identity, "latent_distance"],
    ].rename(columns={"latent_distance": "benign_score"})

    clean_poison = clean.merge(poison, on=identity, validate="one_to_one")
    if clean_poison.empty:
        raise ValueError(
            f"No clean-Nightshade pairs for attack={attack}"
        )
    validate_domainnet_pool(clean_poison, f"Clean-Nightshade pairs for {attack}")

    expected_pairs = len(clean_poison)
    paired = clean_poison.merge(
        processed, on=identity, validate="one_to_one"
    )
    if len(paired) != expected_pairs:
        raise ValueError(
            f"Incomplete pairing for attack={attack}, benign={benign_variant}: "
            f"expected {expected_pairs}, found {len(paired)}"
        )
    validate_domainnet_pool(
        paired,
        f"Clean-Nightshade-benign pairs for {attack}/{benign_variant}",
    )
    score_columns = ["clean_score", "poison_score", "benign_score"]
    if not np.isfinite(paired[score_columns].to_numpy(dtype=float)).all():
        raise ValueError(
            f"Non-finite scores for attack={attack}, benign={benign_variant}"
        )
    return paired.sort_values(identity, kind="mergesort", ignore_index=True)


def selection_masks(
    domains: np.ndarray,
    scenario: str,
    attack_index: int,
):
    for ratio in RATIOS:
        ratio_index = SEED_RATIO_ORDER.index(ratio)
        benign_ratio = ratio if BENIGN_RATIO is None else BENIGN_RATIO
        for repeat in range(REPEATS):
            offset = 0 if scenario == "domain_stratified" else 1_000_000
            rng = np.random.default_rng(
                SEED + offset + attack_index * 10_000 + ratio_index * 100 + repeat
            )
            poison_mask = np.zeros(len(domains), dtype=bool)
            benign_mask = np.zeros(len(domains), dtype=bool)

            if scenario == "domain_stratified":
                domain_candidates = [
                    np.flatnonzero(domains == domain)
                    for domain in sorted(np.unique(domains))
                ]

                # Complete every poison draw first. This preserves the 009
                # Nightshade masks for all domains.
                for candidates in domain_candidates:
                    poison_count = int(round(len(candidates) * ratio))
                    chosen = rng.choice(
                        candidates, size=poison_count, replace=False
                    )
                    poison_mask[chosen] = True

                for candidates in domain_candidates:
                    benign_count = int(
                        round(len(candidates) * benign_ratio)
                    )
                    remaining = candidates[~poison_mask[candidates]]
                    if benign_count > len(remaining):
                        raise ValueError(
                            f"Benign ratio {benign_ratio} exceeds remaining clean pool"
                        )
                    chosen = rng.choice(
                        remaining, size=benign_count, replace=False
                    )
                    benign_mask[chosen] = True
            else:
                poison_count = int(round(len(domains) * ratio))
                chosen = rng.choice(
                    len(domains), size=poison_count, replace=False
                )
                poison_mask[chosen] = True
                remaining = np.flatnonzero(~poison_mask)
                benign_count = int(round(len(domains) * benign_ratio))
                chosen = rng.choice(
                    remaining, size=benign_count, replace=False
                )
                benign_mask[chosen] = True

            if np.any(poison_mask & benign_mask):
                raise AssertionError("Nightshade and benign masks overlap")
            expected_poison_count = int(
                round(EXPECTED_POOL_SIZE * ratio)
            )
            expected_benign_count = int(
                round(EXPECTED_POOL_SIZE * benign_ratio)
            )
            actual_poison_count = int(poison_mask.sum())
            actual_benign_count = int(benign_mask.sum())
            if (
                actual_poison_count != expected_poison_count
                or actual_benign_count != expected_benign_count
            ):
                raise AssertionError(
                    f"Selection count mismatch for {scenario}, ratio={ratio}: "
                    f"expected Nightshade={expected_poison_count}, "
                    f"benign={expected_benign_count}; found "
                    f"Nightshade={actual_poison_count}, "
                    f"benign={actual_benign_count}"
                )
            yield ratio, repeat, poison_mask, benign_mask


def ranking_rows(
    scores: np.ndarray,
    poison_mask: np.ndarray,
    benign_mask: np.ndarray,
    base: dict,
) -> list[dict]:
    labels = poison_mask.astype(np.uint8)
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError(
            "Screening labels contain no positive samples. "
            f"Increase poison_ratio for this sample size: {base}"
        )
    prevalence = float(labels.mean())
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    order = np.argsort(-scores, kind="mergesort")

    rows: list[dict] = []
    for budget in BUDGETS:
        top_k = max(1, int(round(len(scores) * budget)))
        top_indices = order[:top_k]
        nightshade_count = int(poison_mask[top_indices].sum())
        benign_count = int(benign_mask[top_indices].sum())
        untouched_clean_count = top_k - nightshade_count - benign_count
        precision = nightshade_count / top_k
        recall = nightshade_count / positive_count
        rows.append(
            {
                **base,
                "budget": budget,
                "top_k": top_k,
                "auroc": auroc,
                "auprc": auprc,
                "precision_at_top_k": precision,
                "recall_at_top_k": recall,
                "enrichment_at_top_k": precision / prevalence,
                "nightshade_at_top_k": nightshade_count,
                "benign_at_top_k": benign_count,
                "untouched_clean_at_top_k": untouched_clean_count,
                "benign_occupancy_at_top_k": benign_count / top_k,
                "untouched_clean_occupancy_at_top_k": untouched_clean_count
                / top_k,
            }
        )
    return rows


def run_scenario(
    base_scores: pd.DataFrame,
    benign_scores: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for attack_index, attack in enumerate(NIGHTSHADE_ATTACKS):
        for benign_variant in BENIGN_VARIANTS:
            paired = paired_scores(
                base_scores, benign_scores, attack, benign_variant
            )
            domains = paired["domain"].astype(str).to_numpy()
            masks = list(selection_masks(domains, scenario, attack_index))
            clean = paired["clean_score"].to_numpy(dtype=float)
            poison = paired["poison_score"].to_numpy(dtype=float)
            benign = paired["benign_score"].to_numpy(dtype=float)

            for ratio, repeat, poison_mask, benign_mask in masks:
                scores = clean.copy()
                scores[benign_mask] = benign[benign_mask]
                scores[poison_mask] = poison[poison_mask]
                base = {
                    "scenario": scenario,
                    "feature_extractor": FEATURE_EXTRACTOR,
                    "metric": "latent_distance",
                    "probe_transformation": PROBE_TRANSFORMATION,
                    "probe_factor": PROBE_FACTOR,
                    "poisoning": attack[0],
                    "poisoning_factor": attack[1],
                    "benign_transformation": benign_variant[0],
                    "benign_factor": benign_variant[1],
                    "poison_ratio": ratio,
                    "benign_ratio": (
                        ratio if BENIGN_RATIO is None else BENIGN_RATIO
                    ),
                    "repeat": repeat,
                    "pool_size": len(scores),
                    "poison_count": int(poison_mask.sum()),
                    "benign_count": int(benign_mask.sum()),
                    "untouched_clean_count": int(
                        len(scores) - poison_mask.sum() - benign_mask.sum()
                    ),
                }
                rows.extend(
                    ranking_rows(scores, poison_mask, benign_mask, base)
                )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario",
        "feature_extractor",
        "metric",
        "probe_transformation",
        "probe_factor",
        "poisoning",
        "poisoning_factor",
        "benign_transformation",
        "benign_factor",
        "poison_ratio",
        "benign_ratio",
        "pool_size",
        "poison_count",
        "benign_count",
        "untouched_clean_count",
        "budget",
        "top_k",
    ]
    metric_columns = [
        "auroc",
        "auprc",
        "precision_at_top_k",
        "recall_at_top_k",
        "enrichment_at_top_k",
        "nightshade_at_top_k",
        "benign_at_top_k",
        "untouched_clean_at_top_k",
        "benign_occupancy_at_top_k",
        "untouched_clean_occupancy_at_top_k",
    ]
    aggregations: dict[str, tuple] = {
        "repeat_count": ("repeat", "nunique")
    }
    for metric in metric_columns:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_q025"] = (
            metric,
            lambda values: values.quantile(0.025),
        )
        aggregations[f"{metric}_q975"] = (
            metric,
            lambda values: values.quantile(0.975),
        )

    summary = raw.groupby(
        group_columns, as_index=False, dropna=False
    ).agg(**aggregations)
    if not summary["repeat_count"].eq(REPEATS).all():
        counts = sorted(summary["repeat_count"].unique().tolist())
        raise ValueError(f"Expected {REPEATS} repeats per group, found {counts}")
    return summary


def load_or_calculate(
    base_scores: pd.DataFrame,
    benign_scores: pd.DataFrame,
    scenario: str,
    raw_path: Path,
) -> pd.DataFrame:
    if raw_path.is_file() and not OVERWRITE_RAW:
        return pd.read_csv(
            raw_path,
            dtype={
                "probe_factor": "string",
                "poisoning_factor": "string",
                "benign_factor": "string",
            },
        )

    raw = run_scenario(base_scores, benign_scores, scenario)
    raw.to_csv(raw_path, index=False, float_format="%.10g")
    print(f"Saved: {raw_path} ({len(raw)} rows)")
    return raw


def save_summary(raw: pd.DataFrame, path: Path) -> None:
    summary = summarize(raw)
    summary.to_csv(path, index=False, float_format="%.10g")
    print(f"Saved: {path} ({len(summary)} rows)")


def main() -> None:
    SCREENING_ROOT.mkdir(parents=True, exist_ok=True)
    base_scores = load_base_scores()
    benign_scores = load_benign_scores()

    domain = load_or_calculate(
        base_scores,
        benign_scores,
        "domain_stratified",
        DOMAIN_CSV,
    )
    save_summary(domain, DOMAIN_SUMMARY_CSV)

    pooled = load_or_calculate(
        base_scores,
        benign_scores,
        "pooled_random",
        POOLED_CSV,
    )
    save_summary(pooled, POOLED_SUMMARY_CSV)


if __name__ == "__main__":
    main()
