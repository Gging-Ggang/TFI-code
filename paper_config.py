"""Shared datasets, probes, and the fixed source cohort used in the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd


DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
EXCLUDED_CLIPART = frozenset({14, 32, 79, 123, 465})
SOURCE_KEYS = frozenset(
    (domain, index)
    for domain in DOMAINS
    for index in range(500)
    if not (domain == "clipart" and index in EXCLUDED_CLIPART)
)
N_SOURCES = len(SOURCE_KEYS)
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
DATASET_POISONING = {"D_domainnet_clean": ("none", "none")}
for _tag, _strength in (("002", "0.02"), ("004", "0.04")):
    DATASET_POISONING.update({
        f"D_domainnet_nightshade_linf_{_tag}": ("NS_linf", _strength),
        f"D_domainnet_gaussian_linf_{_tag}": ("GN_linf", _strength),
        f"D_domainnet_gaussian_lpips_matched_ns_linf_{_tag}": (
            "GN_star_linf", _strength
        ),
    })
DOMAINNET_DATASETS = tuple(DATASET_POISONING)
AUXILIARY_DOMAINS = {
    "D_Honig_original": "honig", "D_Honig_glaze": "honig",
    "D_Honig_mist": "honig", "D_Honig_antidb": "honig",
    "D_metacloak_clean": "vggface2", "D_metacloak_protected": "vggface2",
}
DATASET_POISONING.update({
    "D_Honig_original": ("none", "none"),
    "D_Honig_glaze": ("Glaze", "none"),
    "D_Honig_mist": ("Honig_MIST", "none"),
    "D_Honig_antidb": ("AntiDB", "none"),
    "D_metacloak_clean": ("none", "none"),
    "D_metacloak_protected": ("MetaCloak", "none"),
})
DATASET_NAMES = tuple(DATASET_POISONING)
AUXILIARY_METHODS = (
    ("MIST", "honig", "Honig_MIST", 90),
    ("Glaze", "honig", "Glaze", 90),
    ("Anti-DreamBooth", "honig", "AntiDB", 90),
    ("MetaCloak", "vggface2", "MetaCloak", 192),
)


def score_column(transformation: str, factor: float) -> str:
    value = str(int(factor)) if transformation == "jpeg_comp" else str(float(factor))
    return f"ld_{transformation}_{value}"


def normalize_identities(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["domain"] = result["domain"].astype(str)
    indices = pd.to_numeric(result["data_index"], errors="raise")
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Source indices must be integers.")
    result["data_index"] = indices.astype(int).map(lambda value: f"{value:04d}")
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


def assert_cohort(data: pd.DataFrame, context: str = "DomainNet") -> None:
    keys = list(zip(data["domain"].astype(str), pd.to_numeric(data["data_index"])))
    actual = set(keys)
    if len(keys) != N_SOURCES or actual != SOURCE_KEYS:
        raise ValueError(
            f"{context}: expected the fixed {N_SOURCES}-source cohort, "
            f"found {len(keys)} rows and {len(actual)} unique identities; "
            f"missing={sorted(SOURCE_KEYS - actual)[:5]}, "
            f"extra={sorted(actual - SOURCE_KEYS)[:5]}"
        )
