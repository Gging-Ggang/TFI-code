from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install it with: pip install scikit-learn"
    ) from exc


# =========================================================
# Hardcoded configuration: identical screening protocol to 014
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_DATASET = "D_domainnet_clean"
NIGHTSHADE_DATASET = "D_domainnet_nightshade_linf_004"
GAUSSIAN_DATASET = "D_domainnet_gaussian_linf_004"
GAUSSIAN_LPIPS_DATASET = "D_domainnet_gaussian_lpips_matched_ns_linf_004"
INPUT_ROOT = PROJECT_ROOT / "result" / "baseline"
OUTPUT_ROOT = INPUT_ROOT / "screening"

SCORE_PATH = INPUT_ROOT / "008_baseline_scores.float32.npy"
ROW_PATH = INPUT_ROOT / "008_baseline_rows.csv"
COLUMN_PATH = INPUT_ROOT / "008_baseline_columns.csv"
MANIFEST_PATH = INPUT_ROOT / "008_baseline_manifest.json"
EXPECTED_TFI_SOURCE = "005_tfi_sd_vae"

DOMAIN_CSV = OUTPUT_ROOT / "008_baseline_screening_domain_stratified.csv"
POOLED_CSV = OUTPUT_ROOT / "008_baseline_screening_pooled_random.csv"
FAIR_CSV = OUTPUT_ROOT / "008_baseline_screening_fair_subset.csv"
DOMAIN_SUMMARY_CSV = (
    OUTPUT_ROOT / "008_baseline_screening_summary_domain_stratified.csv"
)
POOLED_SUMMARY_CSV = (
    OUTPUT_ROOT / "008_baseline_screening_summary_pooled_random.csv"
)
FAIR_SUMMARY_CSV = OUTPUT_ROOT / "008_baseline_screening_summary_fair_subset.csv"

OVERWRITE_RAW = True
RATIOS = (0.10, 0.20)
BUDGETS = (0.01, 0.05, 0.10, 0.20)
REPEATS = 50
SEED = 0
METRICS = ("pixel_rmse", "pixel_mae", "lpips", "frequency_rmse", "tfi")
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


@dataclass(frozen=True)
class AttackSpec:
    poisoning: str
    poisoning_factor: str
    clean_dataset: str
    poison_dataset: str
    cohort: str


MAIN_ATTACKS = (
    AttackSpec(
        "NS_linf",
        "0.04",
        CLEAN_DATASET,
        NIGHTSHADE_DATASET,
        "domainnet",
    ),
    AttackSpec(
        "GN_linf",
        "0.04",
        CLEAN_DATASET,
        GAUSSIAN_DATASET,
        "domainnet",
    ),
    AttackSpec(
        "GN_star_linf",
        "0.04",
        CLEAN_DATASET,
        GAUSSIAN_LPIPS_DATASET,
        "domainnet",
    ),
)

FAIR_ATTACK_NAMES = (
    ("NS_linf", "0.04"),
    ("GN_linf", "0.04"),
    ("GN_star_linf", "0.04"),
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    for path in (SCORE_PATH, ROW_PATH, COLUMN_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing 008 output: {path}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("tfi_source") != EXPECTED_TFI_SOURCE:
        raise ValueError(
            "008 TFI source is missing or incompatible: "
            f"expected '{EXPECTED_TFI_SOURCE}', "
            f"found '{manifest.get('tfi_source')}'"
        )

    rows = pd.read_csv(
        ROW_PATH,
        dtype={"cohort": "string", "domain": "string", "data_index": "string"},
    )
    columns = pd.read_csv(
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
    scores = np.load(SCORE_PATH, mmap_mode="r", allow_pickle=False)

    expected_row_ids = np.arange(len(rows), dtype=np.int64)
    actual_row_ids = rows["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_row_ids, expected_row_ids):
        raise ValueError("008 row_id must be sequential and in score-array row order")

    expected_column_ids = np.arange(len(columns), dtype=np.int64)
    actual_column_ids = columns["column_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_column_ids, expected_column_ids):
        raise ValueError(
            "008 column_id must be sequential and in score-array column order"
        )

    expected_shape = (len(rows), len(columns))
    if scores.shape != expected_shape:
        raise ValueError(f"008 score shape mismatch: {scores.shape} != {expected_shape}")
    if rows.duplicated(["cohort", "domain", "data_index"]).any():
        raise ValueError("008 row metadata contains duplicate identities")
    if columns.drop(columns="column_id").duplicated().any():
        raise ValueError("008 column metadata contains duplicate definitions")

    return rows, columns, scores


def column_id(
    columns: pd.DataFrame,
    dataset_name: str,
    transformation: str,
    transformation_factor: str,
    metric: str,
) -> int:
    selected = columns.loc[
        columns["dataset_name"].eq(dataset_name)
        & columns["transformation"].eq(transformation)
        & columns["transformation_factor"].eq(transformation_factor)
        & columns["metric"].eq(metric),
        "column_id",
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one score column for "
            f"{dataset_name}/{transformation}/{transformation_factor}/{metric}; "
            f"found {len(selected)}"
        )
    return int(selected.iloc[0])


def attack_row_ids(
    rows: pd.DataFrame,
    attack: AttackSpec,
    fair: bool,
) -> np.ndarray:
    selected = rows.loc[rows["cohort"].eq(attack.cohort)].copy()
    if fair:
        fair_domain = "real"
        selected = selected.loc[selected["domain"].eq(fair_domain)]
        expected = 100
        if len(selected) > expected:
            selected = selected.sort_values(
                "data_index",
                kind="mergesort",
            ).iloc[:expected]

    selected = selected.sort_values(
        ["domain", "data_index"],
        kind="mergesort",
    )
    if selected.empty:
        raise ValueError(
            f"No screening rows for {attack.poisoning}/{attack.poisoning_factor}"
        )
    return selected["row_id"].to_numpy(dtype=np.int64)


def selection_masks(
    domains: np.ndarray,
    scenario: str,
    attack_index: int,
):
    for ratio_index, ratio in enumerate(RATIOS):
        for repeat in range(REPEATS):
            if scenario == "domain_stratified":
                offset = 0
            elif scenario == "pooled_random":
                offset = 1_000_000
            elif scenario == "fair_subset":
                offset = 2_000_000
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            rng = np.random.default_rng(
                SEED
                + offset
                + attack_index * 10_000
                + ratio_index * 100
                + repeat
            )
            selected = np.zeros(len(domains), dtype=bool)
            if scenario == "domain_stratified":
                for domain in sorted(np.unique(domains)):
                    candidates = np.flatnonzero(domains == domain)
                    count = int(round(len(candidates) * ratio))
                    selected[
                        rng.choice(candidates, size=count, replace=False)
                    ] = True
            else:
                count = int(round(len(domains) * ratio))
                selected[
                    rng.choice(len(domains), size=count, replace=False)
                ] = True
            yield ratio, repeat, selected


def ranking_rows(
    score_values: np.ndarray,
    labels: np.ndarray,
    base: dict,
) -> list[dict]:
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError(
            "Screening labels contain no positive samples. "
            f"Increase poison_ratio for this sample size: {base}"
        )
    prevalence = float(np.mean(labels))
    auroc = float(roc_auc_score(labels, score_values))
    auprc = float(average_precision_score(labels, score_values))
    order = np.argsort(-score_values, kind="mergesort")
    output: list[dict] = []
    for budget in BUDGETS:
        top_k = max(1, int(round(len(score_values) * budget)))
        true_positive = int(labels[order[:top_k]].sum())
        precision = true_positive / top_k
        recall = true_positive / positive_count
        output.append(
            {
                **base,
                "budget": budget,
                "top_k": top_k,
                "auroc": auroc,
                "auprc": auprc,
                "precision_at_top_k": precision,
                "recall_at_top_k": recall,
                "enrichment_at_top_k": precision / prevalence,
            }
        )
    return output


def run_scenario(
    row_metadata: pd.DataFrame,
    column_metadata: pd.DataFrame,
    score_array: np.ndarray,
    attacks: tuple[AttackSpec, ...],
    scenario: str,
    fair: bool,
) -> pd.DataFrame:
    output: list[dict] = []
    total = len(attacks) * len(METRICS) * len(TRANSFORMATIONS)
    completed = 0

    for attack_index, attack in enumerate(attacks):
        row_ids = attack_row_ids(row_metadata, attack, fair)
        domains = (
            row_metadata.set_index("row_id")
            .loc[row_ids, "domain"]
            .astype(str)
            .to_numpy()
        )
        masks = list(selection_masks(domains, scenario, attack_index))

        for metric in METRICS:
            for transformation, transformation_factor in TRANSFORMATIONS:
                clean_column = column_id(
                    column_metadata,
                    attack.clean_dataset,
                    transformation,
                    transformation_factor,
                    metric,
                )
                poison_column = column_id(
                    column_metadata,
                    attack.poison_dataset,
                    transformation,
                    transformation_factor,
                    metric,
                )
                clean_scores = np.asarray(
                    score_array[row_ids, clean_column],
                    dtype=np.float64,
                )
                poison_scores = np.asarray(
                    score_array[row_ids, poison_column],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(clean_scores)):
                    raise ValueError(
                        f"Non-finite clean scores: {attack.clean_dataset}/"
                        f"{metric}/{transformation}/{transformation_factor}"
                    )
                if not np.all(np.isfinite(poison_scores)):
                    raise ValueError(
                        f"Non-finite poison scores: {attack.poison_dataset}/"
                        f"{metric}/{transformation}/{transformation_factor}"
                    )

                for poison_ratio, repeat, labels in masks:
                    mixed_scores = np.where(labels, poison_scores, clean_scores)
                    base = {
                        "scenario": scenario,
                        "metric": metric,
                        "transformation": transformation,
                        "transformation_factor": transformation_factor,
                        "poisoning": attack.poisoning,
                        "poisoning_factor": attack.poisoning_factor,
                        "clean_dataset": attack.clean_dataset,
                        "poison_dataset": attack.poison_dataset,
                        "poison_ratio": poison_ratio,
                        "repeat": repeat,
                        "pool_size": len(labels),
                        "poison_count": int(labels.sum()),
                    }
                    output.extend(ranking_rows(mixed_scores, labels, base))

                completed += 1
                print(
                    f"[{scenario} {completed}/{total}] "
                    f"{attack.poisoning}/{attack.poisoning_factor} | "
                    f"{metric} | {transformation} {transformation_factor}"
                )

    return pd.DataFrame(output)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario",
        "metric",
        "transformation",
        "transformation_factor",
        "poisoning",
        "poisoning_factor",
        "clean_dataset",
        "poison_dataset",
        "poison_ratio",
        "pool_size",
        "poison_count",
        "budget",
        "top_k",
    ]
    metric_columns = [
        "auroc",
        "auprc",
        "precision_at_top_k",
        "recall_at_top_k",
        "enrichment_at_top_k",
    ]
    aggregations: dict = {"repeat_count": ("repeat", "nunique")}
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
        group_columns,
        as_index=False,
        dropna=False,
    ).agg(**aggregations)
    if not summary["repeat_count"].eq(REPEATS).all():
        counts = sorted(summary["repeat_count"].unique().tolist())
        raise ValueError(f"Expected {REPEATS} repeats per group, found {counts}")
    return summary


def load_or_calculate(
    row_metadata: pd.DataFrame,
    column_metadata: pd.DataFrame,
    score_array: np.ndarray,
    output_path: Path,
    attacks: tuple[AttackSpec, ...],
    scenario: str,
    fair: bool,
) -> pd.DataFrame:
    if output_path.is_file() and not OVERWRITE_RAW:
        print(f"Using existing raw CSV: {output_path}")
        return pd.read_csv(
            output_path,
            dtype={
                "transformation_factor": "string",
                "poisoning_factor": "string",
            },
        )

    raw = run_scenario(
        row_metadata,
        column_metadata,
        score_array,
        attacks,
        scenario,
        fair,
    )
    raw.to_csv(output_path, index=False, float_format="%.10g")
    print(f"Saved: {output_path} ({len(raw)} rows)")
    return raw


def save_summary(raw: pd.DataFrame, output_path: Path) -> None:
    summary = summarize(raw)
    summary.to_csv(output_path, index=False, float_format="%.10g")
    print(f"Saved: {output_path} ({len(summary)} rows)")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, columns, scores = load_inputs()
    fair_lookup = {
        (attack.poisoning, attack.poisoning_factor): attack
        for attack in MAIN_ATTACKS
    }
    missing_fair_attacks = [
        attack for attack in FAIR_ATTACK_NAMES if attack not in fair_lookup
    ]
    if missing_fair_attacks:
        raise ValueError(f"Missing fair attack definitions: {missing_fair_attacks}")
    fair_attacks = tuple(fair_lookup[attack] for attack in FAIR_ATTACK_NAMES)

    domain = load_or_calculate(
        rows,
        columns,
        scores,
        DOMAIN_CSV,
        MAIN_ATTACKS,
        "domain_stratified",
        fair=False,
    )
    save_summary(domain, DOMAIN_SUMMARY_CSV)

    pooled = load_or_calculate(
        rows,
        columns,
        scores,
        POOLED_CSV,
        MAIN_ATTACKS,
        "pooled_random",
        fair=False,
    )
    save_summary(pooled, POOLED_SUMMARY_CSV)

    fair = load_or_calculate(
        rows,
        columns,
        scores,
        FAIR_CSV,
        fair_attacks,
        "fair_subset",
        fair=True,
    )
    save_summary(fair, FAIR_SUMMARY_CSV)


if __name__ == "__main__":
    main()
