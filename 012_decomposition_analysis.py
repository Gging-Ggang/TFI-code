"""Source-paired A, C and gamma at both strengths, without mixture screening."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import (
    CONDITION_NAMES, FEATURE_EXTRACTORS, IDENTITY_COLUMNS,
    STRENGTHS, TRANSFORMATIONS, assert_cohort, domainnet_sources, score_column,
)


# =========================================================
# Hardcoded configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "result" / "features"
TFI_CSV = PROJECT_ROOT / "result" / "tfi" / "tfi.csv"
OUTPUT_CSV = PROJECT_ROOT / "result" / "analysis" / "012_decomposition_per_sample.csv"
FEATURE_DIMS = {
    "sd_vae": 16384, "resnet50": 2048, "convnext_tiny": 768,
    "dinov2_vitb14": 768, "clip_vitl14_336": 768, "vit_b16_384": 768,
}
BATCH_SIZE = 256
EPS = 1e-12


def factor_mask(values: pd.Series, factor: str | float) -> pd.Series:
    if factor == "none":
        return values.astype(str).str.lower().isin(("none", "nan", ""))
    return pd.Series(
        np.isclose(pd.to_numeric(values, errors="coerce"), float(factor)),
        index=values.index,
    )


def load_metadata(feature_root: Path) -> pd.DataFrame:
    path = feature_root / "feature_metadata.parquet"
    metadata = pd.read_parquet(path)
    required = [
        "row_id", "domain", "data_index", "poisoning", "poisoning_factor",
        "transformation", "transformation_factor",
    ]
    missing = set(required) - set(metadata)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not np.array_equal(metadata["row_id"], np.arange(len(metadata))):
        raise ValueError("Metadata row_id must match feature-array row order.")
    for column in required[1:]:
        metadata[column] = metadata[column].astype(str)
    metadata["poisoning"] = metadata["poisoning"].replace(
        {"GN_lpips_matched_ns_linf": "GN_star_linf"}
    )
    return metadata


def indexed_rows(
    metadata: pd.DataFrame, poisoning: str, strength: str | float,
    transformation: str, factor: str | float,
) -> pd.DataFrame:
    mask = (
        metadata["poisoning"].eq(poisoning)
        & factor_mask(metadata["poisoning_factor"], strength)
        & metadata["transformation"].eq(transformation)
        & factor_mask(metadata["transformation_factor"], factor)
    )
    selected = domainnet_sources(metadata.loc[mask, [*IDENTITY_COLUMNS, "row_id"]])
    assert_cohort(selected, f"{poisoning}/{strength}/{transformation}/{factor}")
    return selected.sort_values(IDENTITY_COLUMNS, kind="stable").reset_index(drop=True)


def paired_rows(
    metadata: pd.DataFrame, poisoning: str, strength: float,
    transformation: str, factor: float,
) -> pd.DataFrame:
    settings = (
        ("clean_original_row", "none", "none", "original", "none"),
        ("clean_transformed_row", "none", "none", transformation, factor),
        ("modified_original_row", poisoning, strength, "original", "none"),
        ("modified_transformed_row", poisoning, strength, transformation, factor),
    )
    result = None
    for name, condition, budget, operation, parameter in settings:
        selected = indexed_rows(metadata, condition, budget, operation, parameter)
        if result is None:
            result = selected[IDENTITY_COLUMNS].copy()
        elif not result[IDENTITY_COLUMNS].equals(selected[IDENTITY_COLUMNS]):
            raise ValueError("Four-way source identities must agree exactly.")
        result[name] = selected["row_id"].to_numpy(dtype=np.int64)
    return result


def response_values(
    clean_original: np.ndarray, clean_transformed: np.ndarray,
    modified_original: np.ndarray, modified_transformed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Use the original float32 calculations on the stored float16 features."""
    c = clean_original - clean_transformed
    p = modified_original - clean_original
    q = modified_transformed - clean_transformed
    d = p - q
    modified_response = c + d
    clean_squared = np.einsum("ij,ij->i", c, c, dtype=np.float32, optimize=True)
    modified_squared = np.einsum(
        "ij,ij->i", modified_response, modified_response,
        dtype=np.float32, optimize=True,
    )
    magnitude = np.einsum("ij,ij->i", d, d, dtype=np.float32, optimize=True)
    dot = np.einsum("ij,ij->i", c, d, dtype=np.float32, optimize=True)
    clean_tfi = np.sqrt(clean_squared)
    modified_tfi = np.sqrt(modified_squared)
    denominator = clean_tfi * np.sqrt(magnitude)
    gamma = np.divide(
        dot, denominator, out=np.full_like(dot, np.nan),
        where=(clean_tfi > EPS) & (np.sqrt(magnitude) > EPS),
    )
    gamma = np.clip(gamma, -1.0, 1.0)
    return {
        "clean_tfi": clean_tfi, "modified_tfi": modified_tfi,
        "clean_s2": clean_tfi.astype(np.float64) ** 2,
        "modified_s2": modified_tfi.astype(np.float64) ** 2,
        "A": magnitude, "C": 2.0 * dot, "gamma": gamma,
        "identity_residual": modified_squared - (clean_squared + magnitude + 2.0 * dot),
    }


def existing_scores(
    tfi: pd.DataFrame, extractor: str, poisoning: str, strength: str | float,
    transformation: str, factor: float,
) -> np.ndarray:
    selected = tfi.loc[
        tfi["feature_extractor"].eq(extractor)
        & tfi["poisoning"].eq(poisoning)
        & factor_mask(tfi["poisoning_factor"], strength)
    ]
    selected = domainnet_sources(selected)
    assert_cohort(selected, f"005 TFI {extractor}/{poisoning}/{strength}")
    return selected.sort_values(IDENTITY_COLUMNS, kind="stable")[
        score_column(transformation, factor)
    ].to_numpy(dtype=np.float64)


def calculate(
    metadata: pd.DataFrame, tfi: pd.DataFrame, feature_root: Path, output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(".tmp.csv")
    header = True
    try:
        for extractor in FEATURE_EXTRACTORS:
            path = feature_root / extractor / "features.float16.npy"
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = (len(metadata), FEATURE_DIMS[extractor])
            if features.shape != expected_shape:
                raise ValueError(f"{path}: {features.shape} != {expected_shape}")
            for strength in STRENGTHS:
                for poisoning in ("NS_linf", "GN_star_linf"):
                    for transformation, factor in TRANSFORMATIONS:
                        pairs = paired_rows(
                            metadata, poisoning, strength, transformation, factor,
                        )
                        clean_existing = existing_scores(
                            tfi, extractor, "none", "none", transformation, factor,
                        )
                        modified_existing = existing_scores(
                            tfi, extractor, poisoning, strength, transformation, factor,
                        )
                        for start in range(0, len(pairs), BATCH_SIZE):
                            end = min(start + BATCH_SIZE, len(pairs))
                            rows = pairs.iloc[start:end]
                            arrays = [
                                np.asarray(features[rows[name].to_numpy()], dtype=np.float32)
                                for name in (
                                    "clean_original_row", "clean_transformed_row",
                                    "modified_original_row", "modified_transformed_row",
                                )
                            ]
                            values = response_values(*arrays)
                            if not all(np.isfinite(value).all() for value in values.values()):
                                raise ValueError(
                                    f"Undefined response values: {extractor}/{strength}/"
                                    f"{poisoning}/{transformation}/{factor}"
                                )
                            for name, observed in (
                                ("clean_tfi", clean_existing[start:end]),
                                ("modified_tfi", modified_existing[start:end]),
                            ):
                                if not np.allclose(values[name], observed, rtol=5e-5, atol=5e-5):
                                    raise ValueError("Feature responses disagree with 005 TFI scores.")
                            frame = rows[IDENTITY_COLUMNS].copy()
                            frame["strength"] = strength
                            frame["feature_extractor"] = extractor
                            frame["transformation"] = transformation
                            frame["transformation_factor"] = float(factor)
                            frame["modification"] = CONDITION_NAMES[poisoning]
                            for name, value in values.items():
                                frame[name] = value
                            frame["clean_tfi_abs_error"] = np.abs(
                                values["clean_tfi"] - clean_existing[start:end]
                            )
                            frame["modified_tfi_abs_error"] = np.abs(
                                values["modified_tfi"] - modified_existing[start:end]
                            )
                            frame.to_csv(
                                temporary, mode="w" if header else "a", header=header,
                                index=False, float_format="%.12g",
                            )
                            header = False
                        print(
                            f"[{extractor}] {strength:g} {poisoning} "
                            f"{transformation}/{factor:g}: {len(pairs)} sources",
                            flush=True,
                        )
            del features
        temporary.replace(output_csv)
    except BaseException:
        if temporary.is_file():
            temporary.unlink()
        raise
    print(f"Saved: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--tfi-csv", type=Path, default=TFI_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    metadata = load_metadata(args.feature_root)
    columns = [
        *IDENTITY_COLUMNS, "feature_extractor", "poisoning", "poisoning_factor",
        *(score_column(name, value) for name, value in TRANSFORMATIONS),
    ]
    tfi = pd.read_csv(
        args.tfi_csv, usecols=columns,
        dtype={name: str for name in columns[:5]}, keep_default_na=False,
        float_precision="round_trip",
    )
    tfi["poisoning"] = tfi["poisoning"].replace(
        {"GN_lpips_matched_ns_linf": "GN_star_linf"}
    )
    calculate(metadata, tfi, args.feature_root, args.output_csv)


if __name__ == "__main__":
    main()
