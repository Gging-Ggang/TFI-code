from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except ImportError as exc:
    raise ImportError(
        "scipy is required for KS and Wasserstein calculations. "
        "Install it with: pip install scipy"
    ) from exc


# =========================================================
# Hardcoded configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "result" / "tfi"
INPUT_CSV = INPUT_ROOT / "tfi.csv"
INPUT_NPY = INPUT_ROOT / "tfi.float32.npy"
OUTPUT_ROOT = PROJECT_ROOT / "result" / "ks_wasserstein"
OUTPUT_CSV = OUTPUT_ROOT / "ks_wasserstein.csv"

FEATURE_EXTRACTORS = (
    "sd_vae",
    "clip_vitl14_336",
    "dinov2_vitb14",
    "resnet50",
    "convnext_tiny",
    "vit_b16_384",
)
POISONING_GROUPS = (
    ("none", "none", "Clean"),
    ("NS_linf", "0.04", "Nightshade L-inf 0.04"),
    ("GN_linf", "0.04", "Gaussian noise L-inf 0.04"),
    ("GN_star_linf", "0.04", "LPIPS-matched Gaussian noise L-inf 0.04"),
)

TRANSFORMATIONS = (
    ("downscaling", "0.75", "ld_downscaling_0.75"),
    ("downscaling", "0.6", "ld_downscaling_0.6"),
    ("downscaling", "0.5", "ld_downscaling_0.5"),
    ("downscaling", "0.4", "ld_downscaling_0.4"),
    ("downscaling", "0.3", "ld_downscaling_0.3"),
    ("downscaling", "0.25", "ld_downscaling_0.25"),
    ("gaussian_blur", "0.5", "ld_gaussian_blur_0.5"),
    ("gaussian_blur", "1.0", "ld_gaussian_blur_1.0"),
    ("gaussian_blur", "2.0", "ld_gaussian_blur_2.0"),
    ("gaussian_blur", "3.0", "ld_gaussian_blur_3.0"),
    ("jpeg_comp", "95", "ld_jpeg_comp_95"),
    ("jpeg_comp", "85", "ld_jpeg_comp_85"),
    ("jpeg_comp", "75", "ld_jpeg_comp_75"),
    ("jpeg_comp", "50", "ld_jpeg_comp_50"),
)

LD_COLUMNS = [column for _, _, column in TRANSFORMATIONS]
ANGLE_COLUMNS = [column.replace("ld_", "angle_", 1) for column in LD_COLUMNS]
METRICS = (
    ("latent_distance", LD_COLUMNS),
    ("latent_angle_degree", ANGLE_COLUMNS),
)


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing TFI CSV: {INPUT_CSV}")
    if not INPUT_NPY.is_file():
        raise FileNotFoundError(f"Missing TFI NPY: {INPUT_NPY}")

    metadata_columns = [
        "domain",
        "data_index",
        "feature_extractor",
        "poisoning",
        "poisoning_factor",
    ]
    dataframe = pd.read_csv(
        INPUT_CSV,
        usecols=metadata_columns,
        dtype={column: "string" for column in metadata_columns},
    )
    metrics = np.load(INPUT_NPY, mmap_mode="r", allow_pickle=False)

    all_metric_columns = [*LD_COLUMNS, *ANGLE_COLUMNS]
    expected_shape = (len(dataframe), len(all_metric_columns))
    if metrics.shape != expected_shape:
        raise ValueError(
            f"Expected TFI metric shape {expected_shape}, found {metrics.shape}"
        )
    if not np.issubdtype(metrics.dtype, np.floating):
        raise ValueError(f"TFI metrics must be floating point: {metrics.dtype}")

    dataframe[all_metric_columns] = metrics

    actual_extractors = set(dataframe["feature_extractor"].dropna().astype(str))
    missing_extractors = [
        extractor
        for extractor in FEATURE_EXTRACTORS
        if extractor not in actual_extractors
    ]
    if missing_extractors:
        raise ValueError(f"Missing feature extractors: {missing_extractors}")

    return dataframe


def poisoning_groups(dataframe: pd.DataFrame) -> list[tuple[str, str, str]]:
    observed = [
        tuple(row)
        for row in dataframe.loc[:, ["poisoning", "poisoning_factor"]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    ]
    known = {(poisoning, factor): label for poisoning, factor, label in POISONING_GROUPS}
    ordered = [
        (poisoning, factor, label)
        for poisoning, factor, label in POISONING_GROUPS
        if (poisoning, factor) in observed
    ]
    extras = sorted(group for group in observed if group not in known)
    return [*ordered, *[(poisoning, factor, poisoning) for poisoning, factor in extras]]


def group_mask(
    dataframe: pd.DataFrame,
    extractor_name: str,
    poisoning: str,
    poisoning_factor: str,
) -> np.ndarray:
    return (
        dataframe["feature_extractor"].eq(extractor_name)
        & dataframe["poisoning"].eq(poisoning)
        & dataframe["poisoning_factor"].eq(poisoning_factor)
    ).to_numpy()


def finite_values(
    dataframe: pd.DataFrame,
    mask: np.ndarray,
    column: str,
    context: str,
) -> np.ndarray:
    values = dataframe.loc[mask, column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"No finite values for {context}: {column}")
    return values


def calculate_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    groups = poisoning_groups(dataframe)

    for extractor_name in FEATURE_EXTRACTORS:
        for metric_name, metric_columns in METRICS:
            column_mapping = dict(zip(LD_COLUMNS, metric_columns))

            for transformation, transformation_factor, ld_column in TRANSFORMATIONS:
                value_column = column_mapping[ld_column]
                for poisoning, poisoning_factor, poisoning_label in groups:
                    comparison_mask = group_mask(
                        dataframe,
                        extractor_name,
                        poisoning,
                        poisoning_factor,
                    )
                    comparison_domains = dataframe.loc[
                        comparison_mask,
                        "domain",
                    ].unique()
                    clean_mask = group_mask(
                        dataframe,
                        extractor_name,
                        "none",
                        "none",
                    ) & dataframe["domain"].isin(comparison_domains).to_numpy()
                    clean_values = finite_values(
                        dataframe,
                        clean_mask,
                        value_column,
                        f"{extractor_name}/Clean/{metric_name}/{comparison_domains}",
                    )
                    comparison_values = finite_values(
                        dataframe,
                        comparison_mask,
                        value_column,
                        f"{extractor_name}/{poisoning}/{metric_name}",
                    )
                    ks_result = ks_2samp(
                        clean_values,
                        comparison_values,
                        alternative="two-sided",
                        method="auto",
                    )

                    rows.append(
                        {
                            "feature_extractor": extractor_name,
                            "metric": metric_name,
                            "transformation": transformation,
                            "transformation_factor": transformation_factor,
                            "reference_poisoning": "none",
                            "reference_poisoning_factor": "none",
                            "reference_label": "Clean",
                            "reference_n": len(clean_values),
                            "comparison_poisoning": poisoning,
                            "comparison_poisoning_factor": poisoning_factor,
                            "comparison_label": poisoning_label,
                            "comparison_n": len(comparison_values),
                            "ks_statistic": float(ks_result.statistic),
                            "ks_pvalue": float(ks_result.pvalue),
                            "wasserstein_distance": float(
                                wasserstein_distance(
                                    clean_values,
                                    comparison_values,
                                )
                            ),
                        }
                    )

    results = pd.DataFrame(rows)
    expected_rows = (
        len(FEATURE_EXTRACTORS)
        * len(METRICS)
        * len(TRANSFORMATIONS)
        * len(groups)
    )
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows, found {len(results)}"
        )
    return results


def main() -> None:
    dataframe = load_data()
    results = calculate_results(dataframe)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.10g",
    )

    print(f"Rows: {len(results)}")
    print(f"Saved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
