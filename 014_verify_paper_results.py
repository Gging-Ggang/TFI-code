"""Verify paper results; optionally compare them with independent saved tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_config import N_SOURCES, PROBE_COLUMNS, STRENGTHS


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result" / "paper"
PROBE_KEYS = ["strength", *PROBE_COLUMNS]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def compare_table(
    result_path: Path, reference_path: Path, keys: list[str],
    reference_rename: dict | None = None,
) -> dict:
    actual = read_csv(result_path).set_index(keys, verify_integrity=True)
    expected = read_csv(reference_path).rename(
        columns=reference_rename or {}
    ).set_index(keys, verify_integrity=True)
    missing = expected.index.difference(actual.index)
    if len(missing):
        raise AssertionError(f"{result_path.name}: missing reference keys {list(missing[:5])}")
    actual = actual.loc[expected.index]
    columns = [name for name in expected if name in actual]
    if not columns:
        raise AssertionError(f"{result_path.name}: no comparable reference columns.")
    maximum_error, numeric_values = 0.0, 0
    for name in columns:
        a, b = actual[name], expected[name]
        if pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            if not np.allclose(av, bv, rtol=5e-11, atol=5e-10, equal_nan=True):
                mismatch = ~np.isclose(av, bv, rtol=5e-11, atol=5e-10, equal_nan=True)
                index = int(np.flatnonzero(mismatch)[0])
                raise AssertionError(
                    f"{result_path.name}/{name}/{expected.index[index]}: "
                    f"{av[index]} != {bv[index]}"
                )
            differences = np.abs(av - bv)
            finite_differences = differences[np.isfinite(differences)]
            if len(finite_differences):
                maximum_error = max(maximum_error, float(finite_differences.max()))
            numeric_values += len(b)
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            raise AssertionError(f"{result_path.name}/{name}: reference text differs.")
    return {
        "output": result_path.name, "reference": str(reference_path.resolve()),
        "rows": len(expected), "numeric_values_compared": numeric_values,
        "maximum_absolute_difference": maximum_error,
    }


def verify(root: Path) -> list[dict]:
    checks = []
    probes = read_csv(root / "013_probe_metrics.csv")
    if len(probes) != 168 or probes.duplicated(PROBE_KEYS).any():
        raise AssertionError("Expected 168 unique probes across two strengths.")
    auroc_columns = [name for name in probes if name.startswith("AUROC_")]
    if not probes[auroc_columns].apply(lambda column: column.between(0, 1)).all().all():
        raise AssertionError("AUROCs must be finite and within [0, 1].")
    if not probes["n"].eq(N_SOURCES).all():
        raise AssertionError("All probes must use the fixed 2995-source cohort.")
    for strength, direct_below, gn_below in ((0.02, 23, 3), (0.04, 24, 7)):
        group = probes.loc[probes["strength"].eq(strength)]
        if len(group) != 84:
            raise AssertionError("Each strength must contain all 84 probes.")
        if int(group["AUROC_NS_GNstar"].lt(0.5).sum()) != direct_below:
            raise AssertionError(f"NS-GNstar reversal count differs at {strength}.")
        if int(group["AUROC_NS_GN"].lt(0.5).sum()) != gn_below:
            raise AssertionError(f"NS-GN reversal count differs at {strength}.")
        if group[["AUROC_NS_GN", "AUROC_NS_GNstar"]].eq(0.5).any().any():
            raise AssertionError("The published probe grid has no exact AUROC 0.5 ties.")
        if not group["median_shift_NS_clean"].gt(0).all():
            raise AssertionError("The published NS median shifts must all be positive.")
        checks.append(dict(strength=strength, probes=84, direct_reversals=direct_below))
    group = probes.loc[probes["strength"].eq(0.04)]
    high_clean_reversals = (
        group["AUROC_NS_clean"].ge(0.95) & group["AUROC_NS_GNstar"].lt(0.5)
    ).sum()
    if int(high_clean_reversals) != 5:
        raise AssertionError("Expected five high-clean-separation reversal probes.")
    baseline = read_csv(root / "013_baseline_auroc.csv")
    frequency = baseline.loc[
        baseline["strength"].eq(0.04) & baseline["metric"].eq("frequency_rmse"),
        "AUROC_NS_clean",
    ].item()
    if round(float(frequency), 4) != 0.8090:
        raise AssertionError("Frequency RMSE NS-clean AUROC must round to 0.8090.")
    partial = read_csv(root / "013_partial_spearman.csv")
    for strength, expected in ((0.02, 0.622), (0.04, 0.634)):
        value = partial.loc[
            partial["strength"].eq(strength), "partial_spearman_correlation",
        ].item()
        if round(float(value), 3) != expected:
            raise AssertionError(f"Partial Spearman differs at {strength}.")
    overlap = read_csv(root / "013_top3_overlap.csv")
    if len(overlap) != 2 or not overlap["top3_intersection"].eq(0).all():
        raise AssertionError("Top-three probe intersections must be zero at both strengths.")
    offsets = read_csv(root / "013_source_sign_offsets.csv")
    for strength, opposite, positive_a_negative_s2 in (
        (0.02, 34379, 26167), (0.04, 31241, 26205),
    ):
        row = offsets.loc[offsets["strength"].eq(strength)].iloc[0]
        if (
            int(row["comparisons"]) != N_SOURCES * 84
            or int(row["opposite_signs"]) != opposite
            or int(row["positive_A_negative_s2"]) != positive_a_negative_s2
        ):
            raise AssertionError(f"Source-level sign counts differ at {strength}.")
    checks.append({
        "high_clean_reversals_strength004": 5, "frequency_rmse_auroc_strength004": frequency,
        "partial_spearman": partial.to_dict(orient="records"),
        "top3_intersection": [0, 0],
    })
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    comparisons = {
        "probes": ("013_probe_metrics.csv", PROBE_KEYS, None),
        "baselines": ("013_baseline_auroc.csv", ["strength", "metric"], None),
        "medians": ("013_probe_response_medians.csv", PROBE_KEYS, None),
        "correlations": (
            "013_response_correlations.csv",
            ["strength", "group_scope", "group", "variable", "target"], None,
        ),
        "partial": (
            "013_partial_spearman.csv", ["strength"],
            {"partial_rank_correlation": "partial_spearman_correlation"},
        ),
        "domains": ("013_domain_metrics.csv", ["strength", "domain"], None),
        "auxiliary": ("013_auxiliary_metrics.csv", ["method"], None),
        "benign": ("013_benign_metrics.csv", ["strength", "processing", "processing_factor"], None),
        "paired": ("013_paired_ordering.csv", PROBE_KEYS, None),
        "source-signs": ("013_source_sign_offsets.csv", ["strength"], None),
        "joint-signs": (
            "013_joint_sign_distributions.csv",
            ["strength", "direct_order", "delta_A_sign", "delta_gamma_sign"], None,
        ),
    }
    for name in comparisons:
        parser.add_argument(f"--reference-{name}", type=Path)
    args = parser.parse_args()
    report = {
        "scope": "Calculations from saved measurements; no image generation or model inference.",
        "status": "failed", "checks": [], "reference_comparisons": [],
    }
    try:
        report["checks"] = verify(args.result_root)
        for name, (filename, keys, rename) in comparisons.items():
            reference = getattr(args, f"reference_{name.replace('-', '_')}")
            if reference is not None:
                report["reference_comparisons"].append(compare_table(
                    args.result_root / filename, reference, keys, rename,
                ))
        report["status"] = "passed"
    except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
        report["error"] = str(error)
        raise
    finally:
        path = args.result_root / "014_verification.json"
        if args.result_root.is_dir():
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
