"""Reproduce the paper's AUROCs, response summaries, and scope-check tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr, wasserstein_distance


# =========================================================
# Hardcoded configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_ROOT / "result"
OUTPUT_ROOT = RESULT_ROOT / "paper"
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
EXCLUDED_CLIPART = {14, 32, 79, 123, 465}
N_SOURCES = 2995
STRENGTHS = (0.02, 0.04)
FEATURE_EXTRACTORS = (
    "sd_vae", "resnet50", "convnext_tiny", "dinov2_vitb14",
    "clip_vitl14_336", "vit_b16_384",
)
TRANSFORMATIONS = (
    *(("jpeg_comp", value) for value in (95, 85, 75, 50)),
    *(("gaussian_blur", value) for value in (0.5, 1.0, 2.0, 3.0)),
    *(("downscaling", value) for value in (0.75, 0.6, 0.5, 0.4, 0.3, 0.25)),
)
PROBE_COLUMNS = ["feature_extractor", "transformation", "transformation_factor"]
IDENTITY_COLUMNS = ["domain", "data_index"]
CONDITION_NAMES = {
    "none": "clean", "clean": "clean", "NS_linf": "NS", "NS": "NS",
    "GN_linf": "GN", "GN": "GN", "GN_star_linf": "GNstar",
    "GN_lpips_matched_ns_linf": "GNstar", "GNstar": "GNstar",
}
AUXILIARY_METHODS = (
    ("MIST", "honig", "Honig_MIST", 90),
    ("Glaze", "honig", "Glaze", 90),
    ("Anti-DreamBooth", "honig", "AntiDB", 90),
    ("MetaCloak", "vggface2", "MetaCloak", 192),
)
PROBE_KEYS = ["strength", *PROBE_COLUMNS]
BASELINE_METRICS = ("pixel_rmse", "pixel_mae", "lpips", "frequency_rmse", "tfi")
RESPONSE_VARIABLES = (
    "delta_A", "delta_A_normalized", "delta_C", "delta_C_normalized",
    "delta_gamma", "delta_s2", "delta_s2_normalized",
)


def score_column(transformation: str, factor: float) -> str:
    value = str(int(factor)) if transformation == "jpeg_comp" else str(float(factor))
    return f"ld_{transformation}_{value}"


SCORE_COLUMNS = [score_column(name, value) for name, value in TRANSFORMATIONS]


def normalize_identities(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["domain"] = result["domain"].astype(str)
    result["data_index"] = pd.to_numeric(result["data_index"]).map(
        lambda value: f"{int(value):04d}"
    )
    return result


def exclude_black_sources(data: pd.DataFrame) -> pd.DataFrame:
    result = normalize_identities(data)
    excluded = result["domain"].eq("clipart") & pd.to_numeric(
        result["data_index"]
    ).isin(EXCLUDED_CLIPART)
    return result.loc[~excluded].copy()


def domainnet_sources(data: pd.DataFrame) -> pd.DataFrame:
    result = exclude_black_sources(data)
    return result.loc[result["domain"].isin(DOMAINS)].copy()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path, float_precision="round_trip", keep_default_na=False, **kwargs,
    )


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "compresslevel": 1} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, float_format="%.12g", compression=compression)


def finite(values, context: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{context}: expected nonempty, finite measurements.")
    return result


def auroc(positive, negative) -> float:
    """Full cross-distribution ordering, with half credit for exact ties."""
    a = finite(positive, "AUROC positive distribution")
    b = np.sort(finite(negative, "AUROC negative distribution"))
    lower = np.searchsorted(b, a, side="left")
    upper = np.searchsorted(b, a, side="right")
    return float(np.mean((lower + upper) / (2.0 * len(b))))


def paired_ordering(positive, negative) -> float:
    a = finite(positive, "Paired positive scores")
    b = finite(negative, "Paired negative scores")
    if a.shape != b.shape:
        raise ValueError("Paired scores must have the same shape.")
    return float(np.mean((a > b).astype(float) + 0.5 * (a == b)))


def describe(values) -> dict:
    a = finite(values, "Distribution summary")
    return {
        "n": len(a), "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)), "median": float(np.median(a)),
    }


def comparison(positive, negative) -> dict:
    a = finite(positive, "Modified scores")
    b = finite(negative, "Reference scores")
    return {
        "median_shift": float(np.median(a) - np.median(b)),
        "KS": float(ks_2samp(a, b).statistic),
        "Wasserstein": float(wasserstein_distance(a, b)), "AUROC": auroc(a, b),
    }


def spearman(x, y) -> float:
    a, b = finite(x, "Spearman x"), finite(y, "Spearman y")
    if a.shape != b.shape:
        raise ValueError("Spearman variables must have the same shape.")
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def partial_spearman(x, y, control) -> float:
    """Average ranks, OLS with intercept, then Pearson of the residuals."""
    ranked = [
        rankdata(finite(values, "Partial Spearman variable"), method="average")
        for values in (x, y, control)
    ]
    if len({len(values) for values in ranked}) != 1:
        raise ValueError("Partial Spearman variables must have the same length.")
    design = np.column_stack((np.ones(len(ranked[0])), ranked[2]))
    residuals = [
        values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        for values in ranked[:2]
    ]
    if any(np.std(values) < 1e-12 for values in residuals):
        return float("nan")
    return float(np.corrcoef(*residuals)[0, 1])


def ordered(data: pd.DataFrame, context: str) -> pd.DataFrame:
    if len(data) != N_SOURCES or data.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"{context}: expected {N_SOURCES} distinct source rows.")
    return data.sort_values(IDENTITY_COLUMNS, kind="stable").reset_index(drop=True)


def load_tfi(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = set(
        IDENTITY_COLUMNS + SCORE_COLUMNS
        + ["feature_extractor", "poisoning", "poisoning_factor", "condition", "strength"]
    )
    raw = normalize_identities(read_csv(
        path, usecols=lambda name: name in wanted,
        dtype={name: str for name in (
            "domain", "data_index", "feature_extractor", "poisoning", "poisoning_factor",
        )},
    ))
    missing = set([*IDENTITY_COLUMNS, "feature_extractor", *SCORE_COLUMNS]) - set(raw)
    if missing:
        raise ValueError(f"{path}: missing TFI columns: {sorted(missing)}")
    if "condition" not in raw:
        raw["condition"] = raw["poisoning"].map(CONDITION_NAMES)
        raw["strength"] = pd.to_numeric(raw["poisoning_factor"], errors="coerce")
    raw.loc[raw["condition"].eq("clean"), "strength"] = 0.0
    raw["strength"] = pd.to_numeric(raw["strength"], errors="coerce")
    wide = domainnet_sources(raw)
    mask = wide["condition"].eq("clean") | (
        wide["condition"].isin(("NS", "GN", "GNstar"))
        & wide["strength"].isin(STRENGTHS)
    )
    wide = wide.loc[mask & wide["feature_extractor"].isin(FEATURE_EXTRACTORS)].copy()
    if len(wide) != N_SOURCES * 6 * 7:
        raise ValueError(
            "Expected six extractors, one clean cohort and three controls/"
            f"modifications at both strengths; found {len(wide)} TFI rows."
        )
    return raw, wide


def evaluate_probes(wide: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows, distributions, reference_scores = [], [], []
    for strength in STRENGTHS:
        for extractor in FEATURE_EXTRACTORS:
            selected = wide.loc[wide["feature_extractor"].eq(extractor)]
            clean = ordered(
                selected.loc[selected["condition"].eq("clean")], f"{extractor}/clean",
            )
            groups = {"clean": clean}
            for name in ("NS", "GN", "GNstar"):
                groups[name] = ordered(selected.loc[
                    selected["condition"].eq(name) & selected["strength"].eq(strength)
                ], f"{extractor}/{name}/{strength}")
            if extractor == "sd_vae":
                for name, group in groups.items():
                    frame = group[IDENTITY_COLUMNS + ["ld_jpeg_comp_85"]].rename(
                        columns={"ld_jpeg_comp_85": "tfi_score"}
                    )
                    frame = frame.assign(strength=strength, condition=name)
                    reference_scores.append(frame)
            for transformation, factor in TRANSFORMATIONS:
                column = score_column(transformation, factor)
                scores = {
                    name: finite(group[column], f"{extractor}/{name}/{column}")
                    for name, group in groups.items()
                }
                x = scores["clean"]
                probe = dict(
                    strength=strength, feature_extractor=extractor,
                    transformation=transformation, transformation_factor=float(factor),
                )
                row = dict(
                    **probe, n=N_SOURCES, median_clean_tfi=float(np.median(x)),
                    median_squared_clean_tfi=float(np.median(x * x)),
                )
                for name in ("NS", "GN", "GNstar"):
                    for metric, value in comparison(scores[name], x).items():
                        row[f"{metric}_{name}_clean"] = value
                row["AUROC_NS_GN"] = auroc(scores["NS"], scores["GN"])
                row["AUROC_NS_GNstar"] = auroc(scores["NS"], scores["GNstar"])
                row["paired_NS_exceeds_GNstar"] = paired_ordering(
                    scores["NS"], scores["GNstar"],
                )
                direct = row["AUROC_NS_GNstar"]
                row["direct_order"] = (
                    "above_0.5" if direct > 0.5 else
                    "below_0.5" if direct < 0.5 else "equal_0.5"
                )
                rows.append(row)
                for name, values in scores.items():
                    metrics = {
                        metric: row[f"{metric}_{name}_clean"]
                        for metric in ("median_shift", "KS", "Wasserstein", "AUROC")
                    } if name != "clean" else {}
                    distributions.append(dict(
                        **probe, condition=name, **describe(values),
                        median_clean_tfi=float(np.median(x)), **metrics,
                    ))
    probes = pd.DataFrame(rows)
    save_csv(probes, output / "013_probe_metrics.csv")
    save_csv(pd.DataFrame(distributions), output / "013_score_distributions.csv")
    save_csv(pd.concat(reference_scores, ignore_index=True), output / "013_fig1_scores.csv")
    reference = probes.loc[
        probes["feature_extractor"].eq("sd_vae")
        & probes["transformation"].eq("jpeg_comp")
        & probes["transformation_factor"].eq(85)
    ]
    save_csv(reference, output / "013_reference_metrics.csv")
    save_csv(probes, output / "013_fig2_auroc.csv")
    save_csv(probes[PROBE_KEYS + [
        "paired_NS_exceeds_GNstar", "AUROC_NS_GNstar",
    ]], output / "013_paired_ordering.csv")
    counts, correlations, paired_summaries = [], [], []
    for strength, group in probes.groupby("strength", sort=True):
        for control in ("GN", "GNstar"):
            values = group[f"AUROC_NS_{control}"]
            counts.append(dict(
                strength=strength, control=control, n_probes=len(group),
                above_0_5=int(values.gt(0.5).sum()),
                below_0_5=int(values.lt(0.5).sum()),
                equal_0_5=int(values.eq(0.5).sum()),
                positive_NS_median_shifts=int(group["median_shift_NS_clean"].gt(0).sum()),
                reversed_and_clean_auroc_at_least_0_95=int(
                    (values.lt(0.5) & group["AUROC_NS_clean"].ge(0.95)).sum()
                ),
            ))
        for variable in ("AUROC_NS_clean", "KS_NS_clean", "paired_NS_exceeds_GNstar"):
            correlations.append(dict(
                strength=strength, variable=variable, target="AUROC_NS_GNstar",
                n_probes=len(group), spearman_rho=spearman(
                    group[variable], group["AUROC_NS_GNstar"],
                ),
            ))
        reference = group.loc[
            group["feature_extractor"].eq("sd_vae")
            & group["transformation"].eq("jpeg_comp")
            & group["transformation_factor"].eq(85)
        ].iloc[0]
        paired = group["paired_NS_exceeds_GNstar"]
        direct = group["AUROC_NS_GNstar"]
        paired_summaries.append(dict(
            strength=strength, n_probes=len(group),
            spearman_rho=spearman(paired, direct),
            direction_agreement=int((np.sign(paired - 0.5) == np.sign(direct - 0.5)).sum()),
            median_absolute_difference=float(np.median(np.abs(paired - direct))),
            reference_paired=float(reference["paired_NS_exceeds_GNstar"]),
            reference_auroc=float(reference["AUROC_NS_GNstar"]),
        ))
    save_csv(pd.DataFrame(counts), output / "013_direct_order_counts.csv")
    save_csv(pd.DataFrame(correlations), output / "013_clean_direct_correlations.csv")
    save_csv(pd.DataFrame(paired_summaries), output / "013_paired_ordering_summary.csv")
    return probes


def load_baselines(path: Path | None, baseline_root: Path) -> pd.DataFrame:
    if path is not None:
        return normalize_identities(read_csv(path, dtype={"data_index": str}))
    rows = normalize_identities(read_csv(
        baseline_root / "008_baseline_rows.csv", dtype={"data_index": str},
    ))
    columns = read_csv(
        baseline_root / "008_baseline_columns.csv",
        dtype={"poisoning_factor": str},
    )
    values = np.load(
        baseline_root / "008_baseline_scores.float32.npy", mmap_mode="r", allow_pickle=False,
    )
    if values.shape != (len(rows), len(columns)):
        raise ValueError("Baseline array and row/column manifests disagree.")
    if not np.array_equal(columns["column_id"], np.arange(len(columns))):
        raise ValueError("Baseline column_id must match the array column order.")
    selected = columns.loc[
        columns["transformation"].eq("jpeg_comp")
        & pd.to_numeric(columns["transformation_factor"]).eq(85)
        & columns["metric"].isin(BASELINE_METRICS)
    ].copy()
    selected["condition"] = selected["poisoning"].map(CONDITION_NAMES)
    frames = []
    for strength in STRENGTHS:
        settings = selected.loc[
            selected["condition"].eq("clean") | (
                selected["condition"].isin(("NS", "GNstar"))
                & pd.to_numeric(selected["poisoning_factor"], errors="coerce").eq(strength)
            )
        ]
        for condition, group in settings.groupby("condition", sort=False):
            frame = rows[IDENTITY_COLUMNS].copy()
            for metric in BASELINE_METRICS:
                column = group.loc[group["metric"].eq(metric), "column_id"]
                if len(column) != 1:
                    raise ValueError(f"Missing or duplicate baseline {strength}/{condition}/{metric}")
                frame[metric] = values[:, int(column.iloc[0])]
            frames.append(frame.assign(strength=strength, condition=condition))
    return pd.concat(frames, ignore_index=True)


def evaluate_baselines(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    data = domainnet_sources(data)
    results = []
    for strength in STRENGTHS:
        group = data.loc[data["strength"].eq(strength)]
        scores = {
            name: ordered(group.loc[group["condition"].eq(name)], f"Baseline/{strength}/{name}")
            for name in ("clean", "NS", "GNstar")
        }
        for metric in BASELINE_METRICS:
            row = dict(strength=strength, metric=metric, n_per_condition=N_SOURCES)
            for positive, negative in (("NS", "clean"), ("GNstar", "clean"), ("NS", "GNstar")):
                row[f"AUROC_{positive}_{negative}"] = auroc(
                    scores[positive][metric], scores[negative][metric],
                )
            results.append(row)
    result = pd.DataFrame(results)
    save_csv(result, output / "013_baseline_auroc.csv")
    return result


def evaluate_responses(
    paths: list[Path], probes: pd.DataFrame, output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "strength", *IDENTITY_COLUMNS, *PROBE_COLUMNS, "modification",
        "clean_tfi", "modified_tfi", "clean_s2", "modified_s2", "A", "C", "gamma",
    ]
    frames = [
        read_csv(path, usecols=required, dtype={"data_index": str})
        for path in paths
    ]
    source = domainnet_sources(pd.concat(frames, ignore_index=True))
    source["transformation_factor"] = pd.to_numeric(source["transformation_factor"])
    source["strength"] = pd.to_numeric(source["strength"])
    source["modification"] = source["modification"].map(CONDITION_NAMES)
    if not source["modification"].isin(("NS", "GNstar")).all():
        raise ValueError("Response inputs must contain NS and GNstar measurements only.")
    expected_probes = {
        (extractor, transformation, float(factor))
        for extractor in FEATURE_EXTRACTORS for transformation, factor in TRANSFORMATIONS
    }
    medians, offsets = [], []
    for strength in STRENGTHS:
        selected = source.loc[source["strength"].eq(strength)]
        if len(selected) != N_SOURCES * 84 * 2:
            raise ValueError(f"Response {strength}: expected {N_SOURCES * 84 * 2} rows.")
        for condition in ("NS", "GNstar"):
            group = selected.loc[selected["modification"].eq(condition)]
            found = set(group[PROBE_COLUMNS].itertuples(index=False, name=None))
            if found != expected_probes:
                raise ValueError(f"Response {strength}/{condition}: incomplete probe grid.")
            for key, block in group.groupby(PROBE_COLUMNS, sort=False):
                ordered(block, f"Response {strength}/{condition}/{key}")
            for name in ("A", "C", "gamma", "clean_tfi", "modified_tfi", "modified_s2"):
                finite(group[name], f"Response {strength}/{condition}/{name}")
            if group["A"].lt(0).any() or group["gamma"].abs().gt(1.000001).any():
                raise ValueError("Invalid magnitude or directional-alignment measurement.")
        join = [*IDENTITY_COLUMNS, *PROBE_COLUMNS]
        ns = selected.loc[selected["modification"].eq("NS")]
        control = selected.loc[selected["modification"].eq("GNstar")]
        paired = ns.merge(
            control, on=join, suffixes=("_NS", "_GNstar"),
            validate="one_to_one", how="outer", indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError("NS and GNstar response source identities differ.")
        if not np.allclose(
            paired["clean_tfi_NS"], paired["clean_tfi_GNstar"],
            rtol=1e-7, atol=1e-7,
        ):
            raise ValueError("Paired modifications must use the same clean response.")
        differences = paired[join].copy()
        differences["strength"] = strength
        for column, name in (
            ("A", "delta_A"), ("C", "delta_C"),
            ("gamma", "delta_gamma"), ("modified_s2", "delta_s2"),
        ):
            differences[name] = paired[f"{column}_NS"] - paired[f"{column}_GNstar"]
        median = differences.groupby(PROBE_COLUMNS, sort=False)[
            ["delta_A", "delta_C", "delta_gamma", "delta_s2"]
        ].median().reset_index()
        median = median.assign(strength=strength, paired_n=N_SOURCES)
        medians.append(median)
        a, s2 = differences["delta_A"], differences["delta_s2"]
        offsets.append(dict(
            strength=strength, comparisons=len(differences),
            opposite_signs=int(((a.gt(0) & s2.lt(0)) | (a.lt(0) & s2.gt(0))).sum()),
            positive_A_negative_s2=int((a.gt(0) & s2.lt(0)).sum()),
        ))
        print(f"Source-paired responses {strength:g}: {len(differences)} comparisons", flush=True)
    median = pd.concat(medians, ignore_index=True).merge(
        probes[PROBE_KEYS + [
            "median_squared_clean_tfi", "median_clean_tfi",
            "AUROC_NS_GNstar", "direct_order",
        ]], on=PROBE_KEYS, validate="one_to_one",
    )
    for column in ("delta_A", "delta_C", "delta_s2"):
        median[f"{column}_normalized"] = median[column] / median["median_squared_clean_tfi"]
    save_csv(median, output / "013_probe_response_medians.csv")
    save_csv(median, output / "013_fig3_alignment_auroc.csv")
    save_csv(pd.DataFrame(offsets), output / "013_source_sign_offsets.csv")
    correlations, signs, joint, partial = [], [], [], []
    for strength, group in median.groupby("strength", sort=True):
        scopes = [("all", "all", group)]
        scopes += [
            ("representation", name, block)
            for name, block in group.groupby("feature_extractor", sort=False)
        ]
        scopes += [
            ("transformation_family", name, block)
            for name, block in group.groupby("transformation", sort=False)
        ]
        for scope, name, block in scopes:
            for variable in RESPONSE_VARIABLES:
                correlations.append(dict(
                    strength=strength, group_scope=scope, group=name,
                    variable=variable, target="AUROC_NS_GNstar", n_probes=len(block),
                    spearman_rho=spearman(block[variable], block["AUROC_NS_GNstar"]),
                ))
        partial.append(dict(
            strength=strength, n_probes=len(group),
            partial_spearman_correlation=partial_spearman(
                group["AUROC_NS_GNstar"], group["delta_gamma"],
                group["delta_A_normalized"],
            ),
        ))
        for direction in ("above_0.5", "below_0.5", "equal_0.5"):
            block = group.loc[group["direct_order"].eq(direction)]
            for variable in ("delta_A", "delta_gamma"):
                for sign in (-1, 0, 1):
                    signs.append(dict(
                        strength=strength, direct_order=direction, variable=variable,
                        sign=sign, n_probes=int(np.sign(block[variable]).eq(sign).sum()),
                        group_n=len(block),
                    ))
            for magnitude_sign in (-1, 0, 1):
                for alignment_sign in (-1, 0, 1):
                    joint.append(dict(
                        strength=strength, direct_order=direction,
                        delta_A_sign=magnitude_sign, delta_gamma_sign=alignment_sign,
                        n_probes=int((
                            np.sign(block["delta_A"]).eq(magnitude_sign)
                            & np.sign(block["delta_gamma"]).eq(alignment_sign)
                        ).sum()), group_n=len(block),
                    ))
    partial = pd.DataFrame(partial)
    save_csv(pd.DataFrame(correlations), output / "013_response_correlations.csv")
    save_csv(pd.DataFrame(signs), output / "013_sign_distributions.csv")
    save_csv(pd.DataFrame(joint), output / "013_joint_sign_distributions.csv")
    save_csv(partial, output / "013_partial_spearman.csv")
    return median, partial


def evaluate_domains(wide: pd.DataFrame, output: Path) -> None:
    reference = wide.loc[wide["feature_extractor"].eq("sd_vae")]
    rows = []
    for strength in STRENGTHS:
        for domain in DOMAINS:
            selected = reference.loc[reference["domain"].eq(domain)]
            clean = selected.loc[selected["condition"].eq("clean"), "ld_jpeg_comp_85"]
            row = dict(strength=strength, domain=domain, n=len(clean))
            row.update({f"clean_{key}": value for key, value in describe(clean).items()})
            groups = {}
            for condition in ("NS", "GN", "GNstar"):
                values = selected.loc[
                    selected["condition"].eq(condition) & selected["strength"].eq(strength),
                    "ld_jpeg_comp_85",
                ]
                if len(values) != len(clean):
                    raise ValueError("Domain-wise cohort sizes differ.")
                groups[condition] = values
                for key, value in comparison(values, clean).items():
                    row[f"{key}_{condition}_clean"] = value
                row[f"median_{condition}"] = float(np.median(values))
            row["AUROC_NS_GNstar"] = auroc(groups["NS"], groups["GNstar"])
            rows.append(row)
    save_csv(pd.DataFrame(rows), output / "013_domain_metrics.csv")


def evaluate_auxiliary(raw: pd.DataFrame, output: Path) -> None:
    if "poisoning" not in raw:
        raise ValueError("Auxiliary analysis needs the original 005 TFI CSV, including poisoning labels.")
    rows, distributions = [], []
    for method, domain, poisoning, expected_n in AUXILIARY_METHODS:
        selected = raw.loc[
            raw["domain"].eq(domain) & raw["feature_extractor"].eq("sd_vae")
        ]
        clean = selected.loc[selected["poisoning"].eq("none")]
        modified = selected.loc[selected["poisoning"].eq(poisoning)]
        for condition, group in (("clean", clean), ("modified", modified)):
            if len(group) != expected_n or group.duplicated(IDENTITY_COLUMNS).any():
                raise ValueError(f"{method}/{condition}: expected {expected_n} unique source identities.")
            distributions.append(dict(
                method=method, condition=condition, **describe(group["ld_jpeg_comp_85"]),
            ))
        clean_keys = set(clean[IDENTITY_COLUMNS].itertuples(index=False, name=None))
        modified_keys = set(modified[IDENTITY_COLUMNS].itertuples(index=False, name=None))
        if clean_keys != modified_keys:
            raise ValueError(f"{method}: clean and modified source identities differ.")
        x, y = clean["ld_jpeg_comp_85"], modified["ld_jpeg_comp_85"]
        rows.append(dict(
            method=method, n=expected_n, median_clean=float(np.median(x)),
            median_modified=float(np.median(y)), **comparison(y, x),
        ))
    save_csv(pd.DataFrame(rows), output / "013_auxiliary_metrics.csv")
    save_csv(pd.DataFrame(distributions), output / "013_auxiliary_distributions.csv")


def evaluate_magnitudes(path: Path, output: Path) -> None:
    data = exclude_black_sources(read_csv(path, dtype={"data_index": str, "poisoning_factor": str}))
    data["condition"] = data["poisoning"].map(CONDITION_NAMES)
    data["strength"] = pd.to_numeric(data["poisoning_factor"], errors="coerce")
    rows = []
    for strength in STRENGTHS:
        for condition in ("NS", "GN", "GNstar"):
            selected = data.loc[
                data["domain"].isin(DOMAINS) & data["condition"].eq(condition)
                & data["strength"].eq(strength)
            ]
            ordered(selected, f"Perturbation magnitude/{condition}/{strength}")
            row = dict(strength=strength, condition=condition, n=len(selected))
            for metric in ("linf", "lpips"):
                row.update({
                    f"{metric}_{key}": value
                    for key, value in describe(selected[metric]).items() if key != "n"
                })
            rows.append(row)
    save_csv(pd.DataFrame(rows), output / "013_perturbation_magnitudes.csv")
    rows = []
    for method, domain, poisoning, expected_n in AUXILIARY_METHODS:
        selected = data.loc[data["domain"].eq(domain) & data["poisoning"].eq(poisoning)]
        if len(selected) != expected_n or selected.duplicated(IDENTITY_COLUMNS).any():
            raise ValueError(f"Auxiliary magnitude/{method}: expected {expected_n} unique identities.")
        row = dict(method=method, n=expected_n)
        for metric in ("linf", "lpips"):
            row.update({
                f"{metric}_{key}": value
                for key, value in describe(selected[metric]).items() if key != "n"
            })
        rows.append(row)
    save_csv(pd.DataFrame(rows), output / "013_auxiliary_magnitudes.csv")


def evaluate_benign(path: Path, wide: pd.DataFrame, output: Path) -> None:
    data = domainnet_sources(read_csv(path, dtype={"data_index": str}))
    if "tfi_processed_clean" not in data:
        data = data.rename(columns={"latent_distance": "tfi_processed_clean"})
    required = [
        *IDENTITY_COLUMNS, "benign_transformation", "benign_factor", "tfi_processed_clean",
    ]
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Benign input is missing columns: {sorted(missing)}")
    data = data[required].copy()
    data["benign_factor"] = pd.to_numeric(data["benign_factor"])
    observed = set(data[["benign_transformation", "benign_factor"]].itertuples(index=False, name=None))
    expected = {(name, float(value)) for name, value in TRANSFORMATIONS}
    if observed != expected:
        raise ValueError("Benign input must contain the same 14 operations as the probe grid.")
    reference = wide.loc[wide["feature_extractor"].eq("sd_vae")]
    clean = ordered(reference.loc[reference["condition"].eq("clean")], "Benign/clean reference")
    clean = clean[IDENTITY_COLUMNS + ["ld_jpeg_comp_85"]].rename(
        columns={"ld_jpeg_comp_85": "tfi_untouched_clean"}
    )
    data = data.merge(clean, on=IDENTITY_COLUMNS, validate="many_to_one")
    for strength in STRENGTHS:
        ns = ordered(reference.loc[
            reference["condition"].eq("NS") & reference["strength"].eq(strength)
        ], f"Benign/NS/{strength}")
        tag = f"{int(round(strength * 100)):03d}"
        ns = ns[IDENTITY_COLUMNS + ["ld_jpeg_comp_85"]].rename(
            columns={"ld_jpeg_comp_85": f"tfi_NS_{tag}"}
        )
        data = data.merge(ns, on=IDENTITY_COLUMNS, validate="many_to_one")
    rows = []
    for (operation, factor), group in data.groupby(
        ["benign_transformation", "benign_factor"], sort=True,
    ):
        ordered(group, f"Benign/{operation}/{factor}")
        clean_values = group["tfi_untouched_clean"]
        processed_values = group["tfi_processed_clean"]
        for strength in STRENGTHS:
            ns = group[f"tfi_NS_{int(round(strength * 100)):03d}"]
            row = dict(
                strength=strength, processing=operation, processing_factor=float(factor),
                n=len(group), AUROC_NS_clean=auroc(ns, clean_values),
                AUROC_NS_benign=auroc(ns, processed_values),
                AUROC_benign_clean=auroc(processed_values, clean_values),
                median_shift_benign_clean=float(np.median(processed_values) - np.median(clean_values)),
                KS_benign_clean=float(ks_2samp(processed_values, clean_values).statistic),
            )
            row.update({
                f"processed_{key}": value
                for key, value in describe(processed_values).items() if key != "n"
            })
            rows.append(row)
    save_csv(pd.DataFrame(rows), output / "013_benign_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfi-csv", type=Path, default=RESULT_ROOT / "tfi" / "tfi.csv")
    parser.add_argument(
        "--responses", type=Path, action="append",
        help="Source-response CSV; repeat for separate strength files.",
    )
    parser.add_argument("--baseline-csv", type=Path, help="Optional wide direct-response score CSV.")
    parser.add_argument("--baseline-root", type=Path, default=RESULT_ROOT / "baseline")
    parser.add_argument(
        "--benign-csv", type=Path,
        default=RESULT_ROOT / "tfi" / "010_benign_tfi_sd_vae_jpeg85.csv",
    )
    parser.add_argument(
        "--perturbation-csv", type=Path,
        default=RESULT_ROOT / "perturbation" / "perturbation_per_image.csv",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    responses = args.responses or [
        RESULT_ROOT / "analysis" / "012_decomposition_per_sample.csv"
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw, wide = load_tfi(args.tfi_csv)
    probes = evaluate_probes(wide, args.output_root)
    evaluate_domains(wide, args.output_root)
    evaluate_auxiliary(raw, args.output_root)
    evaluate_baselines(load_baselines(args.baseline_csv, args.baseline_root), args.output_root)
    evaluate_responses(responses, probes, args.output_root)
    evaluate_magnitudes(args.perturbation_csv, args.output_root)
    evaluate_benign(args.benign_csv, wide, args.output_root)
    print(f"Saved paper-analysis outputs: {args.output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
