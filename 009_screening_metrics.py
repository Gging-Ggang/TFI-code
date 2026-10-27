from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as exc:
    raise ImportError("Install scikit-learn with: pip install scikit-learn") from exc


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "result" / "tfi"
OUTPUT_ROOT = ROOT / "result" / "screening"
DOMAIN_CSV = OUTPUT_ROOT / "009_screening_metrics_domain_stratified.csv"
POOLED_CSV = OUTPUT_ROOT / "009_screening_metrics_pooled_random.csv"
FAIR_CSV = OUTPUT_ROOT / "009_screening_metrics_fair_subset.csv"
DOMAIN_SUMMARY_CSV = OUTPUT_ROOT / "009_screening_summary_domain_stratified.csv"
POOLED_SUMMARY_CSV = OUTPUT_ROOT / "009_screening_summary_pooled_random.csv"
FAIR_SUMMARY_CSV = OUTPUT_ROOT / "009_screening_summary_fair_subset.csv"
OVERWRITE_RAW = True
RATIOS = (0.10, 0.20)
BUDGETS = (0.01, 0.05, 0.10, 0.20)
REPEATS = 50
SEED = 0
EXTRACTORS = ("sd_vae", "clip_vitl14_336", "dinov2_vitb14", "resnet50", "convnext_tiny", "vit_b16_384")
MAIN_ATTACKS = (
    ("NS_linf", "0.04"),
    ("GN_linf", "0.04"),
    ("GN_star_linf", "0.04"),
)
FAIR_ATTACKS = MAIN_ATTACKS
TRANSFORMS = (("downscaling", "0.75"), ("downscaling", "0.6"), ("downscaling", "0.5"), ("downscaling", "0.4"), ("downscaling", "0.3"), ("downscaling", "0.25"), ("gaussian_blur", "0.5"), ("gaussian_blur", "1.0"), ("gaussian_blur", "2.0"), ("gaussian_blur", "3.0"), ("jpeg_comp", "95"), ("jpeg_comp", "85"), ("jpeg_comp", "75"), ("jpeg_comp", "50"))
LD = [f"ld_{name}_{factor}" for name, factor in TRANSFORMS]
ANGLE = [c.replace("ld_", "angle_", 1) for c in LD]
METRICS = (("latent_distance", LD), ("latent_angle_degree", ANGLE))


def load() -> pd.DataFrame:
    meta = ["domain", "data_index", "feature_extractor", "poisoning", "poisoning_factor"]
    data = pd.read_csv(INPUT_ROOT / "tfi.csv", usecols=meta, dtype={c: "string" for c in meta})
    array = np.load(INPUT_ROOT / "tfi.float32.npy", mmap_mode="r", allow_pickle=False)
    columns = [*LD, *ANGLE]
    if array.shape != (len(data), len(columns)):
        raise ValueError(f"Metric shape mismatch: {array.shape}")
    data[columns] = array

    return data


def attack_groups(
    data: pd.DataFrame,
    preferred: tuple[tuple[str, str], ...],
    include_extras: bool,
) -> tuple[tuple[str, str], ...]:
    observed = [
        tuple(row)
        for row in data.loc[:, ["poisoning", "poisoning_factor"]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
        if tuple(row) != ("none", "none")
    ]
    ordered = [attack for attack in preferred if attack in observed]
    if not include_extras:
        return tuple(ordered)
    extras = sorted(attack for attack in observed if attack not in preferred)
    return tuple([*ordered, *extras])


def paired_scores(data: pd.DataFrame, extractor: str, attack: tuple[str, str], fair: bool) -> pd.DataFrame:
    poison, factor = attack
    columns = ["domain", "data_index", *LD, *ANGLE]
    clean = data.loc[data["feature_extractor"].eq(extractor) & data["poisoning"].eq("none") & data["poisoning_factor"].eq("none"), columns]
    attack_data = data.loc[data["feature_extractor"].eq(extractor) & data["poisoning"].eq(poison) & data["poisoning_factor"].eq(factor), columns]
    if fair:
        fair_domain = "real"
        clean = clean.loc[clean["domain"].eq(fair_domain)]
        attack_data = attack_data.loc[attack_data["domain"].eq(fair_domain)]
    paired = clean.merge(attack_data, on=["domain", "data_index"], suffixes=("_clean", "_poison"), validate="one_to_one")
    expected = len(paired)
    if fair:
        expected = min(expected, 100)
    if fair and len(paired) > expected:
        paired = paired.sort_values("data_index", kind="mergesort").iloc[:expected].copy()
    if len(paired) == 0:
        raise ValueError(f"No paired rows for {attack}")
    return paired.sort_values(["domain", "data_index"], kind="mergesort", ignore_index=True)


def selection_masks(domains: np.ndarray, scenario: str, attack_index: int):
    for ratio_index, ratio in enumerate(RATIOS):
        for repeat in range(REPEATS):
            offset = 0 if scenario == "domain_stratified" else 1_000_000 if scenario == "pooled_random" else 2_000_000
            rng = np.random.default_rng(SEED + offset + attack_index * 10_000 + ratio_index * 100 + repeat)
            selected = np.zeros(len(domains), dtype=bool)
            if scenario == "domain_stratified":
                for domain in sorted(np.unique(domains)):
                    candidates = np.flatnonzero(domains == domain)
                    count = int(round(len(candidates) * ratio))
                    selected[rng.choice(candidates, size=count, replace=False)] = True
            else:
                count = int(round(len(domains) * ratio))
                selected[rng.choice(len(domains), size=count, replace=False)] = True
            yield ratio, repeat, selected


def ranking_rows(scores: np.ndarray, labels: np.ndarray, base: dict) -> list[dict]:
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError(
            "Screening labels contain no positive samples. "
            f"Increase poison_ratio for this sample size: {base}"
        )
    prevalence = float(np.mean(labels))
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    order = np.argsort(-scores, kind="mergesort")
    rows = []
    for budget in BUDGETS:
        k = max(1, int(round(len(scores) * budget)))
        true_positive = int(labels[order[:k]].sum())
        precision = true_positive / k
        recall = true_positive / positive_count
        rows.append({**base, "budget": budget, "top_k": k, "auroc": auroc, "auprc": auprc, "precision_at_top_k": precision, "recall_at_top_k": recall, "enrichment_at_top_k": precision / prevalence})
    return rows


def run_scenario(data: pd.DataFrame, attacks: tuple[tuple[str, str], ...], scenario: str, fair: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for extractor in EXTRACTORS:
        for attack_index, attack in enumerate(attacks):
            paired = paired_scores(data, extractor, attack, fair)
            domains = paired["domain"].astype(str).to_numpy()
            masks = list(selection_masks(domains, scenario, attack_index))
            for metric, columns in METRICS:
                for (transformation, transformation_factor), column in zip(TRANSFORMS, columns):
                    clean_scores = paired[f"{column}_clean"].to_numpy(dtype=float)
                    poison_scores = paired[f"{column}_poison"].to_numpy(dtype=float)
                    if not np.all(np.isfinite(clean_scores)) or not np.all(np.isfinite(poison_scores)):
                        raise ValueError(f"Non-finite scores: {extractor}/{attack}/{column}")
                    for ratio, repeat, labels in masks:
                        scores = np.where(labels, poison_scores, clean_scores)
                        base = {"scenario": scenario, "feature_extractor": extractor, "metric": metric, "transformation": transformation, "transformation_factor": transformation_factor, "poisoning": attack[0], "poisoning_factor": attack[1], "poison_ratio": ratio, "repeat": repeat, "pool_size": len(labels), "poison_count": int(labels.sum())}
                        rows.extend(ranking_rows(scores, labels, base))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario",
        "feature_extractor",
        "metric",
        "transformation",
        "transformation_factor",
        "poisoning",
        "poisoning_factor",
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
    aggregations = {"repeat_count": ("repeat", "nunique")}
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
    summary = raw.groupby(group_columns, as_index=False, dropna=False).agg(
        **aggregations
    )
    if not summary["repeat_count"].eq(REPEATS).all():
        counts = sorted(summary["repeat_count"].unique().tolist())
        raise ValueError(f"Expected {REPEATS} repeats per group, found {counts}")
    return summary


def load_or_calculate(
    data: pd.DataFrame,
    raw_path: Path,
    attacks: tuple[tuple[str, str], ...],
    scenario: str,
    fair: bool,
) -> pd.DataFrame:
    if raw_path.is_file() and not OVERWRITE_RAW:
        print(f"Using existing raw CSV: {raw_path}")
        return pd.read_csv(
            raw_path,
            dtype={
                "transformation_factor": "string",
                "poisoning_factor": "string",
            },
        )

    raw = run_scenario(data, attacks, scenario, fair)
    raw.to_csv(raw_path, index=False, float_format="%.10g")
    print(f"Saved: {raw_path} ({len(raw)} rows)")
    return raw


def save_summary(raw: pd.DataFrame, path: Path) -> None:
    summary = summarize(raw)
    summary.to_csv(path, index=False, float_format="%.10g")
    print(f"Saved: {path} ({len(summary)} rows)")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load()
    main_attacks = attack_groups(data, MAIN_ATTACKS, include_extras=True)
    fair_attacks = attack_groups(data, FAIR_ATTACKS, include_extras=False)
    domain = load_or_calculate(
        data, DOMAIN_CSV, main_attacks, "domain_stratified", fair=False
    )
    save_summary(domain, DOMAIN_SUMMARY_CSV)
    pooled = load_or_calculate(
        data, POOLED_CSV, main_attacks, "pooled_random", fair=False
    )
    save_summary(pooled, POOLED_SUMMARY_CSV)
    fair = load_or_calculate(
        data, FAIR_CSV, fair_attacks, "fair_subset", fair=True
    )
    save_summary(fair, FAIR_SUMMARY_CSV)


if __name__ == "__main__":
    main()
