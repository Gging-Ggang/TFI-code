from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    import lpips
except ImportError as exc:
    raise ImportError("Install LPIPS with: pip install lpips") from exc

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:
    raise ImportError("Install scikit-image with: pip install scikit-image") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_ROOT / "result"
FEATURE_METADATA = RESULT_ROOT / "features" / "feature_metadata.parquet"
OUTPUT_ROOT = RESULT_ROOT / "perturbation"
PER_IMAGE_CSV = OUTPUT_ROOT / "perturbation_per_image.csv"
SUMMARY_CSV = OUTPUT_ROOT / "perturbation_summary.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024

ATTACKS = (
    ("GN_linf", "0.04"),
    ("NS_linf", "0.04"),
    ("GN_star_linf", "0.04"),
)
METRICS = ("linf", "l2", "l2_per_pixel", "l1_mean_abs", "psnr", "ssim", "lpips")
KEY_COLUMNS = ["domain", "data_index", "poisoning", "poisoning_factor"]


def resolve_stored_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configured_attack_mask(data: pd.DataFrame) -> np.ndarray:
    configured = set(ATTACKS)
    return np.fromiter(
        (
            tuple(row) in configured
            for row in data.loc[:, ["poisoning", "poisoning_factor"]]
            .astype(str)
            .itertuples(index=False, name=None)
        ),
        dtype=bool,
        count=len(data),
    )


def load_pairs() -> pd.DataFrame:
    metadata = pd.read_parquet(FEATURE_METADATA)
    required = ["domain", "data_index", "poisoning", "poisoning_factor", "transformation", "image_path"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing feature metadata columns: {missing}")
    for column in required:
        metadata[column] = metadata[column].astype(str)
    original = metadata.loc[metadata["transformation"].eq("original")].copy()
    clean = original.loc[
        original["poisoning"].eq("none") & original["poisoning_factor"].eq("none"),
        ["domain", "data_index", "image_path"],
    ].rename(columns={"image_path": "clean_image_path"})
    observed_attacks = set(
        original.loc[
            ~(
                original["poisoning"].eq("none")
                & original["poisoning_factor"].eq("none")
            ),
            ["poisoning", "poisoning_factor"],
        ]
        .itertuples(index=False, name=None)
    )
    missing_attacks = [
        attack for attack in ATTACKS if attack not in observed_attacks
    ]
    if missing_attacks:
        raise ValueError(
            f"Configured attacks are missing from feature metadata: {missing_attacks}"
        )

    attacks = original.loc[
        configured_attack_mask(original),
        ["domain", "data_index", "poisoning", "poisoning_factor", "image_path"],
    ]
    pairs = attacks.merge(clean, on=["domain", "data_index"], how="left", validate="many_to_one")
    if pairs["clean_image_path"].isna().any():
        raise ValueError("Some poisoned images have no matching Clean image")
    return pairs.sort_values(
        ["poisoning", "poisoning_factor", "domain", "data_index"],
        kind="mergesort",
        ignore_index=True,
    )


def load_rgb(relative_path: str) -> np.ndarray:
    path = resolve_stored_path(relative_path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def pixel_metrics(clean: np.ndarray, poison: np.ndarray) -> dict[str, float]:
    if clean.shape != poison.shape:
        raise ValueError(f"Image shape mismatch: {clean.shape} != {poison.shape}")
    difference = poison - clean
    absolute = np.abs(difference)
    squared = np.square(difference, dtype=np.float64)
    mse = float(np.mean(squared))
    return {
        "linf": float(np.max(absolute)),
        "l2": float(np.sqrt(np.sum(squared))),
        "l2_per_pixel": float(np.sqrt(mse)),
        "l1_mean_abs": float(np.mean(absolute)),
        "psnr": float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse)),
        "ssim": float(structural_similarity(clean, poison, channel_axis=2, data_range=1.0)),
    }


def lpips_tensor(images: list[np.ndarray]) -> torch.Tensor:
    array = np.stack(images).transpose(0, 3, 1, 2)
    return torch.from_numpy(array).to(DEVICE).mul_(2.0).sub_(1.0)


def calculate_per_image(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, *METRICS])

    model = lpips.LPIPS(net="alex").to(DEVICE).eval()
    rows: list[dict] = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs.iloc[start : start + BATCH_SIZE]
        clean_images = [load_rgb(path) for path in batch["clean_image_path"]]
        poison_images = [load_rgb(path) for path in batch["image_path"]]
        with torch.inference_mode():
            perceptual = model(lpips_tensor(clean_images), lpips_tensor(poison_images))
            perceptual = perceptual.flatten().detach().cpu().numpy()
        for offset, (_, record) in enumerate(batch.iterrows()):
            values = pixel_metrics(clean_images[offset], poison_images[offset])
            rows.append(
                {
                    "domain": record["domain"],
                    "data_index": record["data_index"],
                    "poisoning": record["poisoning"],
                    "poisoning_factor": record["poisoning_factor"],
                    **values,
                    "lpips": float(perceptual[offset]),
                }
            )
        end = min(start + BATCH_SIZE, len(pairs))
        if end % 500 < BATCH_SIZE or end == len(pairs):
            print(f"Processed: {end}/{len(pairs)}")
    result = pd.DataFrame(rows)
    if not np.all(np.isfinite(result[[metric for metric in METRICS if metric != "psnr"]])):
        raise ValueError("Non-finite perturbation metrics were found")
    return result


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    pooled = per_image.assign(domain_scope="pooled")
    by_domain = per_image.assign(domain_scope=per_image["domain"])
    combined = pd.concat([pooled, by_domain], ignore_index=True)
    rows: list[dict] = []
    for keys, group in combined.groupby(
        ["poisoning", "poisoning_factor", "domain_scope"], sort=False
    ):
        row = dict(zip(["poisoning", "poisoning_factor", "domain_scope"], keys))
        row["n"] = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_q90"] = float(np.quantile(values, 0.90))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    existing = None
    if PER_IMAGE_CSV.is_file():
        existing = pd.read_csv(
            PER_IMAGE_CSV,
            dtype={column: "string" for column in KEY_COLUMNS},
        )
        configured = configured_attack_mask(existing)
        removed_count = int((~configured).sum())
        existing = existing.loc[configured].reset_index(drop=True)
        existing_keys = set(
            map(
                tuple,
                existing.loc[:, KEY_COLUMNS]
                .astype(str)
                .itertuples(index=False, name=None),
            )
        )
        keep = [
            tuple(row) not in existing_keys
            for row in pairs.loc[:, KEY_COLUMNS]
            .astype(str)
            .itertuples(index=False, name=None)
        ]
        pairs = pairs.loc[keep].reset_index(drop=True)
        print(f"Existing perturbation rows: {len(existing)}")
        print(f"Removed obsolete perturbation rows: {removed_count}")
        print(f"New perturbation rows: {len(pairs)}")

    new_per_image = calculate_per_image(pairs)
    if existing is not None:
        per_image = pd.concat([existing, new_per_image], ignore_index=True)
    else:
        per_image = new_per_image
    per_image.to_csv(PER_IMAGE_CSV, index=False, float_format="%.10g")

    summary = summarize(per_image)
    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.10g")
    print(f"Saved: {PER_IMAGE_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
