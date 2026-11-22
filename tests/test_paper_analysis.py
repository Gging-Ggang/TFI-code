from __future__ import annotations

import contextlib
import ast
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_config as config


def load_script(stem):
    spec = importlib.util.spec_from_file_location(stem, ROOT / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


analysis = load_script("013_paper_analysis")
responses = load_script("012_decomposition_analysis")
tfi = load_script("005_calculate_tfi")


def load_functions(stem, names, namespace):
    tree = ast.parse((ROOT / f"{stem}.py").read_text(encoding="utf-8"))
    selected = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names),
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), stem, "exec"), namespace)
    return namespace


class OrderingTests(unittest.TestCase):
    def test_exact_ties_receive_half_credit(self):
        self.assertEqual(analysis.auroc([1, 1], [1, 1]), 0.5)
        self.assertEqual(analysis.auroc([1, 2], [1, 3]), 0.375)

    def test_ordering_direction(self):
        self.assertEqual(analysis.auroc([3, 4], [0, 1]), 1.0)
        self.assertEqual(analysis.auroc([0, 1], [3, 4]), 0.0)

    def test_all_pairs_match_brute_force_with_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a, b = rng.integers(0, 5, 7), rng.integers(0, 5, 9)
            differences = a[:, None] - b[None, :]
            expected = np.mean((differences > 0) + 0.5 * (differences == 0))
            self.assertAlmostEqual(analysis.auroc(a, b), expected)
            self.assertAlmostEqual(analysis.auroc(a, b) + analysis.auroc(b, a), 1.0)

    def test_paired_ordering_is_not_full_distribution_auroc(self):
        a, b = [3, 4, 6], [1, 5, 7]
        self.assertAlmostEqual(analysis.paired_ordering(a, b), 1 / 3)
        self.assertAlmostEqual(analysis.auroc(a, b), 4 / 9)

    def test_nonfinite_or_empty_scores_are_rejected(self):
        for invalid in ([], [1, np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                analysis.auroc(invalid, [1, 2])

    def test_partial_spearman_matches_rank_correlation_formula(self):
        x = [1, 1, 2, 4, 5, 6, 7, 8]
        y = [4, 2, 2, 1, 6, 5, 8, 7]
        z = [1, 3, 2, 6, 4, 8, 5, 7]
        xy, xz, yz = (
            spearmanr(a, b).statistic for a, b in ((x, y), (x, z), (y, z))
        )
        expected = (xy - xz * yz) / np.sqrt((1 - xz * xz) * (1 - yz * yz))
        self.assertAlmostEqual(analysis.partial_spearman(x, y, z), expected, places=12)


class CohortTests(unittest.TestCase):
    def test_the_same_five_sources_are_excluded(self):
        raw = pd.DataFrame(
            [(domain, index) for domain in config.DOMAINS for index in range(500)],
            columns=config.IDENTITY_COLUMNS,
        )
        selected = config.domainnet_sources(raw)
        config.assert_cohort(selected)
        self.assertEqual(len(selected), 2995)
        self.assertEqual(selected["domain"].value_counts()["clipart"], 495)
        self.assertTrue(selected.loc[selected["domain"].eq("clipart"), "data_index"].eq("0314").any())

    def test_equal_row_count_does_not_hide_wrong_source_ids(self):
        data = pd.DataFrame(sorted(config.SOURCE_KEYS), columns=config.IDENTITY_COLUMNS)
        data.loc[0, "data_index"] = 500
        with self.assertRaises(ValueError):
            config.assert_cohort(data)

    def test_both_strengths_are_configured(self):
        budgets = {
            value[1] for name, value in config.DATASET_POISONING.items()
            if name in config.DOMAINNET_DATASETS and value[0] != "none"
        }
        self.assertEqual(budgets, {"0.02", "0.04"})
        self.assertEqual(len(config.DOMAINNET_DATASETS), 7)


class FeatureResponseTests(unittest.TestCase):
    def test_squared_score_identity_and_directional_offset(self):
        x = np.array([[1, 0]], dtype=np.float32)
        tx = np.array([[0, 0]], dtype=np.float32)
        mx = np.array([[-1, 1]], dtype=np.float32)
        mtx = tx.copy()
        values = responses.response_values(x, tx, mx, mtx)
        self.assertAlmostEqual(values["A"][0], 5.0)
        self.assertAlmostEqual(values["C"][0], -4.0)
        self.assertAlmostEqual(values["gamma"][0], -2 / np.sqrt(5), places=6)
        self.assertAlmostEqual(values["modified_s2"][0], 2.0, places=6)

    def test_undefined_alignment_is_not_assigned_a_finite_angle(self):
        x = np.ones((1, 2), dtype=np.float32)
        tx = np.zeros((1, 2), dtype=np.float32)
        values = responses.response_values(x, tx, x, tx)
        self.assertTrue(np.isnan(values["gamma"][0]))

    def test_median_is_taken_after_source_pairing(self):
        keys = frozenset({("clipart", index) for index in range(3)})
        measurements, probes = [], []
        for strength in config.STRENGTHS:
            for extractor in config.FEATURE_EXTRACTORS:
                for operation, factor in config.TRANSFORMATIONS:
                    probe = dict(
                        strength=strength, feature_extractor=extractor,
                        transformation=operation, transformation_factor=float(factor),
                    )
                    probes.append(dict(
                        **probe, median_squared_clean_tfi=4.0, median_clean_tfi=2.0,
                        AUROC_NS_GNstar=0.4, direct_order="below_0.5",
                    ))
                    for condition, magnitudes in (
                        ("NS", (1, 2, 101)), ("GNstar", (1, 101, 102)),
                    ):
                        for index, magnitude in enumerate(magnitudes):
                            measurements.append(dict(
                                **probe, domain="clipart", data_index=f"{index:04d}",
                                modification=condition, A=magnitude, C=0.0, gamma=0.0,
                                clean_tfi=2.0, modified_tfi=np.sqrt(4 + magnitude),
                                clean_s2=4.0, modified_s2=4 + magnitude,
                            ))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "responses.csv"
            pd.DataFrame(measurements).to_csv(source, index=False)
            with (
                patch.object(config, "SOURCE_KEYS", keys),
                patch.object(config, "N_SOURCES", 3),
                patch.object(analysis, "N_SOURCES", 3),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                medians, _ = analysis.evaluate_responses(
                    [source], pd.DataFrame(probes), root / "output",
                )
            self.assertTrue(medians["delta_A"].eq(-1).all())
            self.assertTrue(medians["delta_A_normalized"].eq(-0.25).all())
            self.assertNotEqual(-1, np.median([1, 2, 101]) - np.median([1, 101, 102]))


class AuxiliaryCacheTests(unittest.TestCase):
    def metadata(self):
        records = [
            ("D_domainnet_clean", "clipart", "original", "none"),
            ("D_Honig_original", "honig", "original", "none"),
            ("D_domainnet_clean", "clipart", "downscaling", "0.75"),
            ("D_domainnet_clean", "clipart", "jpeg_comp", "85"),
            ("D_Honig_original", "honig", "jpeg_comp", "85"),
        ]
        return pd.DataFrame([
            dict(
                row_id=index, dataset_name=dataset, domain=domain, data_index="0000",
                transformation=operation, transformation_factor=factor,
            )
            for index, (dataset, domain, operation, factor) in enumerate(records)
        ])

    def test_only_nonreference_auxiliary_cells_can_be_unmeasured(self):
        metadata = self.metadata()
        transforms = (
            ("downscaling", "0.75", "ld_downscaling_0.75"),
            ("jpeg_comp", "85", "ld_jpeg_comp_85"),
        )
        with patch.object(tfi, "TRANSFORMATIONS", transforms):
            original = metadata.loc[metadata["transformation"].eq("original")]
            indices = tfi.transformation_row_indices(metadata, original)
            self.assertEqual(indices["ld_downscaling_0.75"].tolist(), [2, -1])
            features = np.array([[1, 2], [2, 1], [2, 2], [3, 2], [2, 3]], dtype=np.float16)
            with contextlib.redirect_stdout(io.StringIO()):
                distances, _ = tfi.calculate_metrics(
                    features, original["row_id"].to_numpy(), indices, "test",
                )
            self.assertEqual(distances[0].tolist(), [1.0, 2.0])
            self.assertTrue(np.isnan(distances[1, 0]))
            self.assertEqual(distances[1, 1], 2.0)
            for missing_row in (2, 4):
                with self.assertRaises(ValueError):
                    tfi.transformation_row_indices(
                        metadata.loc[metadata["row_id"].ne(missing_row)], original,
                    )


class PipelineConfigurationTests(unittest.TestCase):
    def test_feature_locations_cover_two_strengths_and_reference_only_auxiliary(self):
        namespace = load_functions(
            "004_feature_extraction", {"factor_text", "dataset_locations", "parse_poisoning"},
            {
                "Path": Path, "DatasetLocation": SimpleNamespace,
                "DATA_ROOT": Path("data"), "TRANSFORMATION_ROOT": Path("data/Transformation"),
                "DATASET_NAMES": config.DATASET_NAMES,
                "DOMAINNET_DATASETS": config.DOMAINNET_DATASETS,
                "DATASET_POISONING": config.DATASET_POISONING,
                "AUXILIARY_DOMAINS": config.AUXILIARY_DOMAINS,
                "DOWNSCALE_FACTORS": (.75, .6, .5, .4, .3, .25),
                "GAUSSIAN_SIGMAS": (.5, 1., 2., 3.), "JPEG_QUALITIES": (95, 85, 75, 50),
            },
        )
        with patch.object(Path, "is_dir", return_value=True):
            locations = namespace["dataset_locations"]()
        self.assertEqual(len(locations), 13 + 7 * 14 + 6)
        for location in locations:
            if location.dataset_name in config.AUXILIARY_DOMAINS:
                self.assertIn(
                    (location.transformation, location.transformation_factor),
                    (("original", "none"), ("jpeg_comp", "85")),
                )
        self.assertEqual(
            namespace["parse_poisoning"]("D_domainnet_nightshade_linf_002"),
            ("NS_linf", "0.02"),
        )
        self.assertEqual(
            namespace["parse_poisoning"]("D_domainnet_gaussian_lpips_matched_ns_linf_004"),
            ("GN_star_linf", "0.04"),
        )

    def test_reference_baseline_preserves_005_array_column_order(self):
        native = tuple(
            SimpleNamespace(transformation=name, transformation_factor=factor)
            for name, factor, _ in tfi.TRANSFORMATIONS
        )
        jpeg85 = next(
            item for item in native
            if item.transformation == "jpeg_comp" and item.transformation_factor == "85"
        )
        namespace = load_functions(
            "008_calculate_baseline_screening_scores", {"tfi_scores"},
            {"np": np, "pd": pd, "TFI_TRANSFORMATIONS": native, "TRANSFORMATIONS": (jpeg85,)},
        )
        metadata = pd.DataFrame([dict(
            poisoning="NS_linf", poisoning_factor="0.04",
            domain="clipart", data_index="0000", metric_row=0,
        )])
        scores = namespace["tfi_scores"](
            metadata, np.arange(28, dtype=np.float32).reshape(1, 28),
            SimpleNamespace(poisoning="NS_linf", poisoning_factor="0.04"),
            jpeg85, [("clipart", "0000")],
        )
        self.assertEqual(scores.tolist(), [11.0])


if __name__ == "__main__":
    unittest.main()
