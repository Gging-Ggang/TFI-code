"""Calculate source-paired feature responses at both perturbation strengths."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Hardcoded configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "result" / "features"
OUTPUT_CSV = PROJECT_ROOT / "result" / "analysis" / "012_decomposition_per_sample.csv"
FEATURE_DIMS = {
    "sd_vae": 16384, "resnet50": 2048, "convnext_tiny": 768,
    "dinov2_vitb14": 768, "clip_vitl14_336": 768, "vit_b16_384": 768,
}
BATCH_SIZE = 256
EPS = 1e-12
FEATURE_EXTRACTORS = tuple(FEATURE_DIMS)
IDENTITY_COLUMNS = ["domain", "data_index"]
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
EXCLUDED_CLIPART = {14, 32, 79, 123, 465}
STRENGTHS = (0.02, 0.04)
TRANSFORMATIONS = (
    *(("jpeg_comp", value) for value in (95, 85, 75, 50)),
    *(("gaussian_blur", value) for value in (0.5, 1.0, 2.0, 3.0)),
    *(("downscaling", value) for value in (0.75, 0.6, 0.5, 0.4, 0.3, 0.25)),
)


def domainnet_sources(data: pd.DataFrame) -> pd.DataFrame:
    result = data.loc[data["domain"].isin(DOMAINS)].copy()
    indices = pd.to_numeric(result["data_index"])
    result["data_index"] = indices.map(lambda value: f"{int(value):04d}")
    excluded = result["domain"].eq("clipart") & indices.isin(EXCLUDED_CLIPART)
    return result.loc[~excluded].copy()


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
    if len(selected) != 2995 or selected.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"Incomplete feature rows: {poisoning}/{strength}/{transformation}/{factor}")
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
    }


def calculate(
    metadata: pd.DataFrame, feature_root: Path, output_csv: Path,
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
                            frame = rows[IDENTITY_COLUMNS].copy()
                            frame["strength"] = strength
                            frame["feature_extractor"] = extractor
                            frame["transformation"] = transformation
                            frame["transformation_factor"] = float(factor)
                            frame["modification"] = (
                                "NS" if poisoning == "NS_linf" else "GNstar"
                            )
                            for name, value in values.items():
                                frame[name] = value
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
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    metadata = load_metadata(args.feature_root)
    calculate(metadata, args.feature_root, args.output_csv)


if __name__ == "__main__":
    main()
