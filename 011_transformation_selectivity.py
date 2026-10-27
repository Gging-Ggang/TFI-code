from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
TFI_ROOT = ROOT / "result" / "tfi"
KS_ROOT = ROOT / "result" / "ks_wasserstein"
SCREEN_ROOT = ROOT / "result" / "screening"
OUTPUT_CSV = ROOT / "result" / "analysis" / "011_transformation_selectivity.csv"
EXPECTED_DOMAINNET_POOL_SIZE = 60
DOMAINNET_ATTACKS = {
    ("NS_linf", "0.04"),
    ("GN_linf", "0.04"),
    ("GN_star_linf", "0.04"),
}


def domainnet_attack_mask(data: pd.DataFrame) -> np.ndarray:
    return np.fromiter(
        (
            tuple(row) in DOMAINNET_ATTACKS
            for row in data.loc[:, ["poisoning", "poisoning_factor"]]
            .astype(str)
            .itertuples(index=False, name=None)
        ),
        dtype=bool,
        count=len(data),
    )


def validate_no_domainnet_mist(data: pd.DataFrame, context: str) -> None:
    obsolete = data["poisoning"].astype(str).eq("MIST")
    if obsolete.any():
        raise ValueError(
            f"{context} still contains obsolete DomainNet MIST rows; "
            "regenerate its upstream result first"
        )


def tfi_summary() -> pd.DataFrame:
    metadata_columns = [
        "domain",
        "data_index",
        "feature_extractor",
        "poisoning",
        "poisoning_factor",
    ]
    data = pd.read_csv(
        TFI_ROOT / "tfi.csv",
        usecols=metadata_columns,
        dtype={column: "string" for column in metadata_columns},
    )
    transform_columns = [
        column for column in pd.read_csv(TFI_ROOT / "tfi.csv", nrows=0).columns
        if column.startswith("ld_")
    ]
    values = np.load(TFI_ROOT / "tfi.float32.npy", mmap_mode="r", allow_pickle=False)
    if values.shape[0] != len(data) or values.shape[1] < len(transform_columns):
        raise ValueError(
            f"TFI CSV/NPY mismatch: {len(data)} rows, {values.shape} values"
        )
    data[transform_columns] = values[:, : len(transform_columns)]
    clean = data.loc[
        data["poisoning"].eq("none") & data["poisoning_factor"].eq("none")
    ]
    attacks = data.loc[domainnet_attack_mask(data)]
    rows: list[dict] = []
    for column in transform_columns:
        _, transformation, factor = column.split("_", maxsplit=2)
        if transformation == "gaussian":
            transformation = "gaussian_blur"
            factor = factor.removeprefix("blur_")
        elif transformation == "jpeg":
            transformation = "jpeg_comp"
            factor = factor.removeprefix("comp_")
        clean_values = clean.loc[:, ["feature_extractor", "domain", "data_index", column]]
        for keys, group in attacks.groupby(
            ["feature_extractor", "poisoning", "poisoning_factor"],
            sort=False,
        ):
            extractor, poisoning, poisoning_factor = keys
            paired = clean_values.loc[
                clean_values["feature_extractor"].eq(extractor),
                ["domain", "data_index", column],
            ].merge(
                group.loc[:, ["domain", "data_index", column]],
                on=["domain", "data_index"],
                suffixes=("_clean", "_poison"),
                validate="one_to_one",
            )
            rows.append(
                {
                    "feature_extractor": extractor,
                    "metric": "latent_distance",
                    "transformation": transformation,
                    "transformation_factor": factor,
                    "poisoning": poisoning,
                    "poisoning_factor": poisoning_factor,
                    "clean_n": len(paired),
                    "poison_n": len(paired),
                    "clean_mean": float(paired[f"{column}_clean"].mean()),
                    "poison_mean": float(paired[f"{column}_poison"].mean()),
                    "delta_mean": float(
                        paired[f"{column}_poison"].mean()
                        - paired[f"{column}_clean"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def screening_summary(path: Path, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        dtype={"transformation_factor": "string", "poisoning_factor": "string"},
    )
    validate_no_domainnet_mist(data, str(path))
    if prefix != "fair_subset":
        domainnet = data.loc[domainnet_attack_mask(data)]
        if domainnet.empty or not domainnet["pool_size"].eq(
            EXPECTED_DOMAINNET_POOL_SIZE
        ).all():
            counts = sorted(domainnet["pool_size"].unique().tolist())
            raise ValueError(
                f"{path} does not use the expected outlier-filtered "
                f"DomainNet pool size {EXPECTED_DOMAINNET_POOL_SIZE}: {counts}"
            )
    keys = ["feature_extractor", "metric", "transformation", "transformation_factor", "poisoning", "poisoning_factor"]
    ranking = data.groupby(keys, as_index=False).agg(**{f"{prefix}_auroc": ("auroc", "mean"), f"{prefix}_auprc": ("auprc", "mean")})
    budget = data.groupby([*keys, "budget"], as_index=False).agg(precision=("precision_at_top_k", "mean"), recall=("recall_at_top_k", "mean"), enrichment=("enrichment_at_top_k", "mean"))
    pieces = [ranking]
    for value in sorted(budget["budget"].unique()):
        selected = budget.loc[np.isclose(budget["budget"], value)].drop(columns="budget")
        suffix = str(int(round(value * 100)))
        selected = selected.rename(columns={"precision": f"{prefix}_precision_top_{suffix}pct", "recall": f"{prefix}_recall_top_{suffix}pct", "enrichment": f"{prefix}_enrichment_top_{suffix}pct"})
        pieces.append(selected)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=keys, how="left", validate="one_to_one")
    return result


def severity_value(transformation: str, factor: str) -> float:
    value = float(factor)
    if transformation == "downscaling":
        return 1.0 - value
    if transformation == "gaussian_blur":
        return value
    if transformation == "jpeg_comp":
        return 100.0 - value
    raise ValueError(f"Unknown transformation: {transformation}")


def main() -> None:
    summary = tfi_summary()
    ks = pd.read_csv(KS_ROOT / "ks_wasserstein.csv", dtype={"transformation_factor": "string", "comparison_poisoning_factor": "string"})
    ks = ks.rename(columns={"comparison_poisoning": "poisoning", "comparison_poisoning_factor": "poisoning_factor"})
    validate_no_domainnet_mist(summary, "TFI pooled summary")
    validate_no_domainnet_mist(ks, "006 KS/Wasserstein")
    summary_domainnet = summary.loc[domainnet_attack_mask(summary)]
    ks_domainnet = ks.loc[domainnet_attack_mask(ks)]
    if summary_domainnet.empty or not (
        summary_domainnet["clean_n"].eq(EXPECTED_DOMAINNET_POOL_SIZE)
        & summary_domainnet["poison_n"].eq(EXPECTED_DOMAINNET_POOL_SIZE)
    ).all():
        raise ValueError(
            "TFI pooled summary was not generated with the "
            f"{EXPECTED_DOMAINNET_POOL_SIZE}-image DomainNet pool"
        )
    if ks_domainnet.empty or not (
        ks_domainnet["reference_n"].eq(EXPECTED_DOMAINNET_POOL_SIZE)
        & ks_domainnet["comparison_n"].eq(EXPECTED_DOMAINNET_POOL_SIZE)
    ).all():
        raise ValueError(
            "006 KS/Wasserstein was not generated with the "
            f"{EXPECTED_DOMAINNET_POOL_SIZE}-image DomainNet pool"
        )
    keys = ["feature_extractor", "metric", "transformation", "transformation_factor", "poisoning", "poisoning_factor"]
    ks = ks.loc[:, [*keys, "ks_statistic", "ks_pvalue", "wasserstein_distance"]]
    result = summary.merge(ks, on=keys, how="left", validate="one_to_one")

    for path, prefix in (
        (SCREEN_ROOT / "009_screening_metrics_domain_stratified.csv", "domain_stratified"),
        (SCREEN_ROOT / "009_screening_metrics_pooled_random.csv", "pooled_random"),
        (SCREEN_ROOT / "009_screening_metrics_fair_subset.csv", "fair_subset"),
    ):
        screen = screening_summary(path, prefix)
        result = result.merge(screen, on=keys, how="left", validate="one_to_one")

    result["severity"] = [severity_value(t, f) for t, f in zip(result["transformation"], result["transformation_factor"])]
    result["clean_ld_baseline"] = result["clean_mean"]
    result["poison_clean_gap"] = result["delta_mean"]
    result["primary_auroc"] = result["domain_stratified_auroc"].combine_first(
        result["fair_subset_auroc"]
    )
    result["primary_auprc"] = result["domain_stratified_auprc"].combine_first(
        result["fair_subset_auprc"]
    )
    group_keys = ["feature_extractor", "metric", "poisoning", "poisoning_factor", "transformation"]
    result["ks_rank_within_family"] = result.groupby(group_keys)["ks_statistic"].rank(method="min", ascending=False)
    result["gap_rank_within_family"] = result.groupby(group_keys)["poison_clean_gap"].rank(method="min", ascending=False)
    result["auroc_rank_within_family"] = result.groupby(group_keys)["primary_auroc"].rank(method="min", ascending=False)

    correlation_rows = []
    for keys_value, group in result.dropna(subset=["primary_auroc"]).groupby(group_keys, sort=False):
        ks_corr = spearmanr(group["clean_ld_baseline"], group["ks_statistic"])
        auc_corr = spearmanr(group["clean_ld_baseline"], group["primary_auroc"])
        correlation_rows.append((*keys_value, float(ks_corr.statistic), float(auc_corr.statistic)))
    correlation = pd.DataFrame(correlation_rows, columns=[*group_keys, "clean_baseline_vs_ks_spearman", "clean_baseline_vs_auroc_spearman"])
    result = result.merge(correlation, on=group_keys, how="left", validate="many_to_one")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, float_format="%.10g")
    print(f"Saved: {OUTPUT_CSV} ({len(result)} rows)")


if __name__ == "__main__":
    main()
