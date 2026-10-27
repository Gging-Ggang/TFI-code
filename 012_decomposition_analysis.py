from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =========================================================
# Hardcoded configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "result" / "features"
FEATURE_METADATA = FEATURE_ROOT / "feature_metadata.parquet"
TFI_ROOT = PROJECT_ROOT / "result" / "tfi"
SCREENING_SUMMARY_CSV = (
    PROJECT_ROOT
    / "result"
    / "screening"
    / "009_screening_summary_domain_stratified.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "result" / "analysis"

PER_SAMPLE_CSV = OUTPUT_ROOT / "012_decomposition_per_sample.csv"
SUMMARY_CSV = OUTPUT_ROOT / "012_decomposition_ns_vs_gnstar_summary.csv"
REPRESENTATIVE_CSV = OUTPUT_ROOT / "012_decomposition_representative_probes.csv"
CORRELATION_CSV = OUTPUT_ROOT / "012_decomposition_correlations.csv"
SIGN_COUNTS_CSV = OUTPUT_ROOT / "012_decomposition_sign_counts.csv"

BATCH_SIZE = 256
EPS = 1e-12

# The submitted 60-image sample uses 10% poison prevalence in script 009.
# Set this to 0.05 when reproducing the full-data paper setting.
POISON_RATIO = 0.10
INSPECTION_BUDGET = 0.05

FEATURE_EXTRACTORS = (
    "sd_vae",
    "clip_vitl14_336",
    "dinov2_vitb14",
    "resnet50",
    "convnext_tiny",
    "vit_b16_384",
)
FEATURE_DIMS = {
    "sd_vae": 16384,
    "clip_vitl14_336": 768,
    "dinov2_vitb14": 768,
    "resnet50": 2048,
    "convnext_tiny": 768,
    "vit_b16_384": 768,
}

ATTACKS = (
    ("NS_linf", "0.04"),
    ("GN_star_linf", "0.04"),
    ("GN_linf", "0.04"),
)
PRIMARY_ATTACK = ("NS_linf", "0.04")
CONTROL_ATTACK = ("GN_star_linf", "0.04")

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
LD_COLUMNS = [f"ld_{name}_{factor}" for name, factor in TRANSFORMATIONS]

REPRESENTATIVE_PROBES = (
    ("sd_vae", "jpeg_comp", "85", "SDVAE-J85"),
    ("dinov2_vitb14", "jpeg_comp", "85", "DINO-J85"),
    ("convnext_tiny", "jpeg_comp", "85", "ConvNeXt-J85"),
    ("clip_vitl14_336", "jpeg_comp", "85", "CLIP-J85"),
    ("vit_b16_384", "jpeg_comp", "85", "ViT-J85"),
    ("resnet50", "jpeg_comp", "85", "ResNet-J85"),
    ("sd_vae", "gaussian_blur", "0.5", "SDVAE-Blur0.5"),
)

IDENTITY_COLUMNS = ["domain", "data_index"]
FEATURE_KEY_COLUMNS = [
    "domain",
    "data_index",
    "poisoning",
    "poisoning_factor",
    "transformation",
    "transformation_factor",
]
OUTPUT_COLUMNS = [
    "domain",
    "data_index",
    "feature_extractor",
    "transformation",
    "transformation_factor",
    "poisoning",
    "poisoning_factor",
    "clean_tfi",
    "perturbed_tfi",
    "p_norm",
    "q_norm",
    "delta_norm",
    "A",
    "B",
    "E",
    "retention",
    "pq_cosine",
    "gamma",
    "identity_residual",
    "clean_tfi_existing",
    "perturbed_tfi_existing",
    "clean_tfi_abs_error",
    "perturbed_tfi_abs_error",
]
SUMMARY_QUANTITIES = (
    "A",
    "B",
    "E",
    "delta_norm",
    "retention",
    "pq_cosine",
    "gamma",
)


def load_metadata() -> pd.DataFrame:
    if not FEATURE_METADATA.is_file():
        raise FileNotFoundError(f"Missing feature metadata: {FEATURE_METADATA}")

    required_columns = ["row_id", *FEATURE_KEY_COLUMNS]
    metadata = pd.read_parquet(FEATURE_METADATA, columns=required_columns)
    missing_columns = [
        column for column in required_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Feature metadata is missing columns: {missing_columns}")

    expected_row_ids = np.arange(len(metadata), dtype=np.int64)
    actual_row_ids = metadata["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_row_ids, expected_row_ids):
        raise ValueError("Feature metadata row_id must be sequential from 0 to N-1")

    for column in FEATURE_KEY_COLUMNS:
        metadata[column] = metadata[column].astype("string")

    duplicated = metadata.duplicated(FEATURE_KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = metadata.loc[duplicated, FEATURE_KEY_COLUMNS].head(10)
        raise ValueError(
            "Feature metadata rows are not unique by decomposition key:\n"
            + examples.to_string(index=False)
        )

    expected_attacks = pd.MultiIndex.from_tuples(
        ATTACKS,
        names=["poisoning", "poisoning_factor"],
    )
    observed_attacks = pd.MultiIndex.from_frame(
        metadata.loc[
            ~metadata["poisoning"].eq("none"),
            ["poisoning", "poisoning_factor"],
        ].drop_duplicates()
    )
    missing_attacks = expected_attacks.difference(observed_attacks)
    if len(missing_attacks):
        raise ValueError(f"Configured attacks are missing: {list(missing_attacks)}")

    return metadata


def indexed_rows(
    metadata: pd.DataFrame,
    poisoning: str,
    poisoning_factor: str,
    transformation: str,
    transformation_factor: str,
) -> pd.DataFrame:
    selected = metadata.loc[
        metadata["poisoning"].eq(poisoning)
        & metadata["poisoning_factor"].eq(poisoning_factor)
        & metadata["transformation"].eq(transformation)
        & metadata["transformation_factor"].eq(transformation_factor),
        [*IDENTITY_COLUMNS, "row_id"],
    ].copy()
    if selected.empty:
        raise ValueError(
            "No feature rows for "
            f"{poisoning}/{poisoning_factor}/"
            f"{transformation}/{transformation_factor}"
        )

    duplicated = selected.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicated.any():
        examples = selected.loc[duplicated, IDENTITY_COLUMNS].head(10)
        raise ValueError(
            "Duplicate image identities for "
            f"{poisoning}/{poisoning_factor}/"
            f"{transformation}/{transformation_factor}:\n"
            + examples.to_string(index=False)
        )

    return selected.sort_values(
        IDENTITY_COLUMNS,
        kind="mergesort",
        ignore_index=True,
    )


def paired_rows(
    metadata: pd.DataFrame,
    poisoning: str,
    poisoning_factor: str,
    transformation: str,
    transformation_factor: str,
) -> pd.DataFrame:
    components = (
        ("clean_original_row", "none", "none", "original", "none"),
        (
            "clean_transformed_row",
            "none",
            "none",
            transformation,
            transformation_factor,
        ),
        (
            "poison_original_row",
            poisoning,
            poisoning_factor,
            "original",
            "none",
        ),
        (
            "poison_transformed_row",
            poisoning,
            poisoning_factor,
            transformation,
            transformation_factor,
        ),
    )
    indexed = [
        indexed_rows(metadata, attack, factor, transform, transform_factor).rename(
            columns={"row_id": row_name}
        )
        for row_name, attack, factor, transform, transform_factor in components
    ]

    result = indexed[0]
    for component in indexed[1:]:
        result = result.merge(
            component,
            on=IDENTITY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )

    expected = max(len(component) for component in indexed)
    if len(result) != expected:
        raise ValueError(
            f"Incomplete four-way pairing for {poisoning}/{poisoning_factor} "
            f"{transformation}/{transformation_factor}: "
            f"paired={len(result)}, expected={expected}"
        )
    return result.sort_values(
        IDENTITY_COLUMNS,
        kind="mergesort",
        ignore_index=True,
    )


def load_feature_array(extractor: str, metadata_rows: int) -> np.ndarray:
    feature_path = FEATURE_ROOT / extractor / "features.float16.npy"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing feature array: {feature_path}")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (metadata_rows, FEATURE_DIMS[extractor])
    if features.shape != expected_shape:
        raise ValueError(
            f"Feature shape mismatch for {extractor}: "
            f"expected {expected_shape}, found {features.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"Feature array is not floating point: {feature_path}")
    return features


def load_existing_tfi() -> pd.DataFrame:
    metadata_path = TFI_ROOT / "tfi.csv"
    values_path = TFI_ROOT / "tfi.float32.npy"
    metadata_columns = [
        "domain",
        "data_index",
        "feature_extractor",
        "poisoning",
        "poisoning_factor",
    ]
    data = pd.read_csv(
        metadata_path,
        usecols=metadata_columns,
        dtype={column: "string" for column in metadata_columns},
    )
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(data), len(LD_COLUMNS) * 2):
        raise ValueError(
            f"TFI CSV/NPY shape mismatch: "
            f"expected {(len(data), len(LD_COLUMNS) * 2)}, found {values.shape}"
        )
    data[LD_COLUMNS] = values[:, : len(LD_COLUMNS)]
    return data


def existing_tfi_lookup(
    tfi: pd.DataFrame,
    extractor: str,
    poisoning: str,
    poisoning_factor: str,
    transformation: str,
    transformation_factor: str,
) -> pd.DataFrame:
    column = f"ld_{transformation}_{transformation_factor}"
    selected = tfi.loc[
        tfi["feature_extractor"].eq(extractor)
        & tfi["poisoning"].eq(poisoning)
        & tfi["poisoning_factor"].eq(poisoning_factor),
        [*IDENTITY_COLUMNS, column],
    ].copy()
    selected = selected.rename(columns={column: "existing_tfi"})
    duplicated = selected.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicated.any():
        examples = selected.loc[duplicated, IDENTITY_COLUMNS].head(10)
        raise ValueError(
            f"Duplicate TFI rows for {extractor}/{poisoning}/{poisoning_factor}:\n"
            + examples.to_string(index=False)
        )
    return selected


def calculate_decomposition(
    features: np.ndarray,
    pairs: pd.DataFrame,
    extractor: str,
    poisoning: str,
    poisoning_factor: str,
    transformation: str,
    transformation_factor: str,
    tfi: pd.DataFrame,
) -> pd.DataFrame:
    clean_existing = existing_tfi_lookup(
        tfi,
        extractor,
        "none",
        "none",
        transformation,
        transformation_factor,
    )
    poison_existing = existing_tfi_lookup(
        tfi,
        extractor,
        poisoning,
        poisoning_factor,
        transformation,
        transformation_factor,
    )
    clean_tfi_existing = pairs.loc[:, IDENTITY_COLUMNS].merge(
        clean_existing,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )["existing_tfi"].to_numpy(dtype=np.float32)
    perturbed_tfi_existing = pairs.loc[:, IDENTITY_COLUMNS].merge(
        poison_existing,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )["existing_tfi"].to_numpy(dtype=np.float32)
    if (
        np.isnan(clean_tfi_existing).any()
        or np.isnan(perturbed_tfi_existing).any()
    ):
        raise ValueError(
            "Missing existing TFI rows for "
            f"{extractor}/{poisoning}/{poisoning_factor}/"
            f"{transformation}/{transformation_factor}"
        )

    n = len(pairs)
    values = np.empty((n, 14), dtype=np.float32)
    clean_original_rows = pairs["clean_original_row"].to_numpy(dtype=np.int64)
    clean_transformed_rows = pairs["clean_transformed_row"].to_numpy(dtype=np.int64)
    poison_original_rows = pairs["poison_original_row"].to_numpy(dtype=np.int64)
    poison_transformed_rows = pairs["poison_transformed_row"].to_numpy(dtype=np.int64)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        clean_original = np.asarray(
            features[clean_original_rows[start:end]], dtype=np.float32
        )
        clean_transformed = np.asarray(
            features[clean_transformed_rows[start:end]], dtype=np.float32
        )
        poison_original = np.asarray(
            features[poison_original_rows[start:end]], dtype=np.float32
        )
        poison_transformed = np.asarray(
            features[poison_transformed_rows[start:end]], dtype=np.float32
        )

        c = clean_original - clean_transformed
        p = poison_original - clean_original
        q = poison_transformed - clean_transformed
        delta = p - q
        perturbed_response = c + delta

        clean_tfi_squared = np.einsum(
            "ij,ij->i", c, c, dtype=np.float32, optimize=True
        )
        perturbed_tfi_squared = np.einsum(
            "ij,ij->i",
            perturbed_response,
            perturbed_response,
            dtype=np.float32,
            optimize=True,
        )
        p_squared = np.einsum(
            "ij,ij->i", p, p, dtype=np.float32, optimize=True
        )
        q_squared = np.einsum(
            "ij,ij->i", q, q, dtype=np.float32, optimize=True
        )
        a_values = np.einsum(
            "ij,ij->i", delta, delta, dtype=np.float32, optimize=True
        )
        c_dot_delta = np.einsum(
            "ij,ij->i", c, delta, dtype=np.float32, optimize=True
        )
        pq_dot = np.einsum(
            "ij,ij->i", p, q, dtype=np.float32, optimize=True
        )

        clean_tfi = np.sqrt(clean_tfi_squared)
        perturbed_tfi = np.sqrt(perturbed_tfi_squared)
        p_norm = np.sqrt(p_squared)
        q_norm = np.sqrt(q_squared)
        delta_norm = np.sqrt(a_values)
        b_values = 2.0 * c_dot_delta
        e_values = a_values + b_values
        retention = np.divide(
            q_norm,
            p_norm,
            out=np.full_like(q_norm, np.nan),
            where=p_norm > EPS,
        )
        pq_cosine = np.divide(
            pq_dot,
            p_norm * q_norm,
            out=np.full_like(pq_dot, np.nan),
            where=(p_norm > EPS) & (q_norm > EPS),
        )
        gamma = np.divide(
            c_dot_delta,
            clean_tfi * delta_norm,
            out=np.full_like(c_dot_delta, np.nan),
            where=(clean_tfi > EPS) & (delta_norm > EPS),
        )
        pq_cosine = np.clip(pq_cosine, -1.0, 1.0)
        gamma = np.clip(gamma, -1.0, 1.0)
        identity_residual = perturbed_tfi_squared - (
            clean_tfi_squared + a_values + b_values
        )

        values[start:end] = np.column_stack(
            [
                clean_tfi,
                perturbed_tfi,
                p_norm,
                q_norm,
                delta_norm,
                a_values,
                b_values,
                e_values,
                retention,
                pq_cosine,
                gamma,
                identity_residual,
                np.abs(clean_tfi - clean_tfi_existing[start:end]),
                np.abs(perturbed_tfi - perturbed_tfi_existing[start:end]),
            ]
        )

    if not np.all(np.isfinite(values[:, :12])):
        raise ValueError(
            "Non-finite decomposition values for "
            f"{extractor}/{poisoning}/{poisoning_factor}/"
            f"{transformation}/{transformation_factor}"
        )

    result = pairs.loc[:, IDENTITY_COLUMNS].copy()
    result["feature_extractor"] = extractor
    result["transformation"] = transformation
    result["transformation_factor"] = transformation_factor
    result["poisoning"] = poisoning
    result["poisoning_factor"] = poisoning_factor
    result["clean_tfi"] = values[:, 0]
    result["perturbed_tfi"] = values[:, 1]
    result["p_norm"] = values[:, 2]
    result["q_norm"] = values[:, 3]
    result["delta_norm"] = values[:, 4]
    result["A"] = values[:, 5]
    result["B"] = values[:, 6]
    result["E"] = values[:, 7]
    result["retention"] = values[:, 8]
    result["pq_cosine"] = values[:, 9]
    result["gamma"] = values[:, 10]
    result["identity_residual"] = values[:, 11]
    result["clean_tfi_existing"] = clean_tfi_existing
    result["perturbed_tfi_existing"] = perturbed_tfi_existing
    result["clean_tfi_abs_error"] = values[:, 12]
    result["perturbed_tfi_abs_error"] = values[:, 13]
    return result.loc[:, OUTPUT_COLUMNS]


def write_per_sample(metadata: pd.DataFrame, tfi: pd.DataFrame) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_csv = PER_SAMPLE_CSV.with_suffix(".tmp.csv")
    if temporary_csv.exists():
        temporary_csv.unlink()

    pair_cache = {
        (poisoning, poisoning_factor, transformation, transformation_factor): (
            paired_rows(
                metadata,
                poisoning,
                poisoning_factor,
                transformation,
                transformation_factor,
            )
        )
        for poisoning, poisoning_factor in ATTACKS
        for transformation, transformation_factor in TRANSFORMATIONS
    }

    header = True
    for extractor in FEATURE_EXTRACTORS:
        features = load_feature_array(extractor, len(metadata))
        for poisoning, poisoning_factor in ATTACKS:
            for transformation, transformation_factor in TRANSFORMATIONS:
                pairs = pair_cache[
                    (
                        poisoning,
                        poisoning_factor,
                        transformation,
                        transformation_factor,
                    )
                ]
                chunk = calculate_decomposition(
                    features,
                    pairs,
                    extractor,
                    poisoning,
                    poisoning_factor,
                    transformation,
                    transformation_factor,
                    tfi,
                )
                chunk.to_csv(
                    temporary_csv,
                    mode="w" if header else "a",
                    header=header,
                    index=False,
                    float_format="%.10g",
                )
                header = False
                print(
                    f"[{extractor}] {poisoning}/{poisoning_factor} "
                    f"{transformation}/{transformation_factor}: {len(chunk)} rows"
                )
        del features

    temporary_csv.replace(PER_SAMPLE_CSV)
    print(f"Saved: {PER_SAMPLE_CSV}")


def p5_margins() -> pd.DataFrame:
    data = pd.read_csv(
        SCREENING_SUMMARY_CSV,
        dtype={
            "transformation_factor": "string",
            "poisoning_factor": "string",
        },
    )
    selected = data.loc[
        data["metric"].eq("latent_distance")
        & np.isclose(data["poison_ratio"].astype(float), POISON_RATIO)
        & np.isclose(data["budget"].astype(float), INSPECTION_BUDGET),
        [
            "feature_extractor",
            "transformation",
            "transformation_factor",
            "poisoning",
            "poisoning_factor",
            "precision_at_top_k_mean",
        ],
    ].copy()
    if selected.empty:
        raise ValueError(
            "No screening rows for latent distance with "
            f"poison_ratio={POISON_RATIO} and budget={INSPECTION_BUDGET}"
        )

    keys = ["feature_extractor", "transformation", "transformation_factor"]
    primary = selected.loc[
        selected["poisoning"].eq(PRIMARY_ATTACK[0])
        & selected["poisoning_factor"].eq(PRIMARY_ATTACK[1]),
        [*keys, "precision_at_top_k_mean"],
    ].rename(columns={"precision_at_top_k_mean": "p5_ns"})
    control = selected.loc[
        selected["poisoning"].eq(CONTROL_ATTACK[0])
        & selected["poisoning_factor"].eq(CONTROL_ATTACK[1]),
        [*keys, "precision_at_top_k_mean"],
    ].rename(columns={"precision_at_top_k_mean": "p5_gnstar"})
    result = primary.merge(control, on=keys, how="outer", validate="one_to_one")
    result["p5_margin"] = result["p5_ns"] - result["p5_gnstar"]

    expected_probes = len(FEATURE_EXTRACTORS) * len(TRANSFORMATIONS)
    if len(result) != expected_probes or result.isna().any().any():
        raise ValueError(
            f"Expected {expected_probes} complete P@5 probe rows, found {len(result)}"
        )
    return result


def summarize_pair(data: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "domain",
        "data_index",
        "feature_extractor",
        "transformation",
        "transformation_factor",
    ]
    group_keys = [
        "feature_extractor",
        "transformation",
        "transformation_factor",
    ]
    primary = data.loc[
        data["poisoning"].eq(PRIMARY_ATTACK[0])
        & data["poisoning_factor"].eq(PRIMARY_ATTACK[1]),
        [*keys, *SUMMARY_QUANTITIES],
    ]
    control = data.loc[
        data["poisoning"].eq(CONTROL_ATTACK[0])
        & data["poisoning_factor"].eq(CONTROL_ATTACK[1]),
        [*keys, *SUMMARY_QUANTITIES],
    ]
    paired = primary.merge(
        control,
        on=keys,
        how="inner",
        suffixes=("_ns", "_gnstar"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("No paired NS/GN* decomposition rows were found")

    rows: list[dict] = []
    for group_values, group in paired.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, group_values))
        row["paired_n"] = len(group)
        for quantity in SUMMARY_QUANTITIES:
            ns = group[f"{quantity}_ns"].to_numpy(dtype=float)
            gnstar = group[f"{quantity}_gnstar"].to_numpy(dtype=float)
            difference = ns - gnstar
            row[f"{quantity}_ns_median"] = float(np.nanmedian(ns))
            row[f"{quantity}_ns_q95"] = float(np.nanquantile(ns, 0.95))
            row[f"{quantity}_gnstar_median"] = float(np.nanmedian(gnstar))
            row[f"{quantity}_gnstar_q95"] = float(
                np.nanquantile(gnstar, 0.95)
            )
            row[f"delta_{quantity}_median"] = float(np.nanmedian(difference))
        rows.append(row)

    summary = pd.DataFrame(rows).merge(
        margins,
        on=group_keys,
        how="left",
        validate="one_to_one",
    )
    expected_probes = len(FEATURE_EXTRACTORS) * len(TRANSFORMATIONS)
    if len(summary) != expected_probes or summary["p5_margin"].isna().any():
        raise ValueError(
            f"Expected {expected_probes} complete probe summaries, "
            f"found {len(summary)}"
        )
    return summary.sort_values(
        group_keys,
        kind="mergesort",
        ignore_index=True,
    )


def representative_summary(summary: pd.DataFrame) -> pd.DataFrame:
    probes = pd.DataFrame(
        REPRESENTATIVE_PROBES,
        columns=[
            "feature_extractor",
            "transformation",
            "transformation_factor",
            "probe",
        ],
    )
    result = probes.merge(
        summary,
        on=["feature_extractor", "transformation", "transformation_factor"],
        how="left",
        validate="one_to_one",
    )
    selected_columns = [
        "probe",
        "p5_margin",
        "delta_A_median",
        "delta_B_median",
        "delta_E_median",
        "delta_retention_median",
        "delta_pq_cosine_median",
        "delta_gamma_median",
        "paired_n",
    ]
    if result["p5_margin"].isna().any():
        raise ValueError("One or more representative probes are missing")
    return result.loc[:, selected_columns]


def correlation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    scopes = [("all", summary)]
    scopes.extend(
        (transformation, group)
        for transformation, group in summary.groupby("transformation", sort=False)
    )
    rows = []
    for scope, group in scopes:
        correlation = spearmanr(
            group["delta_gamma_median"],
            group["p5_margin"],
        )
        rows.append(
            {
                "scope": scope,
                "probe_n": len(group),
                "spearman_rho": float(correlation.statistic),
                "spearman_pvalue": float(correlation.pvalue),
            }
        )
    return pd.DataFrame(rows)


def sign_count_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for margin_sign, mask in (
        ("positive", summary["p5_margin"] > 0),
        ("negative", summary["p5_margin"] < 0),
        ("zero", summary["p5_margin"] == 0),
    ):
        selected = summary.loc[mask]
        rows.append(
            {
                "p5_margin_sign": margin_sign,
                "probe_n": len(selected),
                "delta_A_positive_n": int((selected["delta_A_median"] > 0).sum()),
                "delta_A_negative_n": int((selected["delta_A_median"] < 0).sum()),
                "delta_gamma_positive_n": int(
                    (selected["delta_gamma_median"] > 0).sum()
                ),
                "delta_gamma_negative_n": int(
                    (selected["delta_gamma_median"] < 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(data: pd.DataFrame) -> None:
    max_abs_residual = float(data["identity_residual"].abs().max())
    max_clean_error = float(data["clean_tfi_abs_error"].max())
    max_perturbed_error = float(data["perturbed_tfi_abs_error"].max())
    print(f"Max |identity residual|: {max_abs_residual:.6g}")
    print(f"Max clean TFI error vs 005: {max_clean_error:.6g}")
    print(f"Max perturbed TFI error vs 005: {max_perturbed_error:.6g}")


def main() -> None:
    metadata = load_metadata()
    tfi = load_existing_tfi()
    write_per_sample(metadata, tfi)

    decomposition = pd.read_csv(
        PER_SAMPLE_CSV,
        dtype={
            "domain": "string",
            "data_index": "string",
            "feature_extractor": "string",
            "transformation": "string",
            "transformation_factor": "string",
            "poisoning": "string",
            "poisoning_factor": "string",
        },
    )
    validate_outputs(decomposition)

    summary = summarize_pair(decomposition, p5_margins())
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.10g")
    print(f"Saved: {SUMMARY_CSV} ({len(summary)} rows)")

    representative = representative_summary(summary)
    representative.to_csv(REPRESENTATIVE_CSV, index=False, float_format="%.10g")
    print(f"Saved: {REPRESENTATIVE_CSV} ({len(representative)} rows)")

    correlations = correlation_summary(summary)
    correlations.to_csv(CORRELATION_CSV, index=False, float_format="%.10g")
    print(f"Saved: {CORRELATION_CSV} ({len(correlations)} rows)")

    sign_counts = sign_count_summary(summary)
    sign_counts.to_csv(SIGN_COUNTS_CSV, index=False)
    print(f"Saved: {SIGN_COUNTS_CSV} ({len(sign_counts)} rows)")


if __name__ == "__main__":
    main()
