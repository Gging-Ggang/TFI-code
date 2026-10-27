from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from path_config import (
    CLEAN_DATASET,
    DATA_ROOT,
    GAUSSIAN_LPIPS_DATASET,
    NIGHTSHADE_DATASET,
    RESULT_ROOT,
    store_path,
)

try:
    import lpips
except ImportError as exc:
    raise ImportError("Install LPIPS with: pip install lpips") from exc


PERTURBATION_CSV = RESULT_ROOT / "perturbation" / "perturbation_per_image.csv"
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INDEX_RE = re.compile(r"(?P<index>\d{4})$")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Pair:
    domain: str
    data_index: str
    clean_path: Path
    nightshade_path: Path
    output_path: Path


@dataclass(frozen=True)
class TargetSpec:
    poisoning: str
    factor: str
    nightshade_dataset: str
    output_dataset: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a per-image LPIPS-matched Gaussian baseline."
    )
    parser.add_argument("--clean-dataset", default=CLEAN_DATASET)
    parser.add_argument(
        "--targets",
        default=(
            f"0.04:{NIGHTSHADE_DATASET}:{GAUSSIAN_LPIPS_DATASET}"
        ),
        help=(
            "Comma-separated factor:nightshade_dataset:output_dataset specs. "
            "By default this creates the 0.04 matched folder."
        ),
    )
    parser.add_argument("--nightshade-dataset", default=None)
    parser.add_argument("--output-dataset", default=None)
    parser.add_argument("--nightshade-poisoning", default="NS_linf")
    parser.add_argument("--nightshade-factor", default=None)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-alpha", type=float, default=0.5)
    parser.add_argument("--coarse-steps", type=int, default=32)
    parser.add_argument("--refine-steps", type=int, default=9)
    parser.add_argument("--refine-rounds", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-source",
        choices=("auto", "csv", "compute"),
        default="auto",
        help="Use existing perturbation CSV targets when available, or compute LPIPS.",
    )
    return parser.parse_args()


def parse_target_specs(args: argparse.Namespace) -> list[TargetSpec]:
    legacy_values = [
        args.nightshade_dataset,
        args.output_dataset,
        args.nightshade_factor,
    ]
    if any(value is not None for value in legacy_values):
        if not all(value is not None for value in legacy_values):
            raise ValueError(
                "For a single target, provide all of --nightshade-dataset, "
                "--output-dataset, and --nightshade-factor."
            )
        return [
            TargetSpec(
                poisoning=args.nightshade_poisoning,
                factor=args.nightshade_factor,
                nightshade_dataset=args.nightshade_dataset,
                output_dataset=args.output_dataset,
            )
        ]

    specs: list[TargetSpec] = []
    for raw_spec in args.targets.split(","):
        parts = raw_spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Each --targets entry must be "
                "factor:nightshade_dataset:output_dataset"
            )
        factor, nightshade_dataset, output_dataset = parts
        specs.append(
            TargetSpec(
                poisoning=args.nightshade_poisoning,
                factor=factor,
                nightshade_dataset=nightshade_dataset,
                output_dataset=output_dataset,
            )
        )
    return specs


def infer_index(path: Path) -> str:
    match = INDEX_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer four-digit data index: {path}")
    return match.group("index")


def images_by_index(root: Path, domain: str) -> dict[str, Path]:
    domain_root = root / domain
    if not domain_root.is_dir():
        raise FileNotFoundError(f"Missing domain directory: {domain_root}")
    result: dict[str, Path] = {}
    for path in sorted(domain_root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index = infer_index(path)
            if index in result:
                raise ValueError(f"Duplicate index {domain}/{index}: {result[index]} and {path}")
            result[index] = path
    return result


def collect_pairs(clean_root: Path, nightshade_root: Path, output_root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for domain in DOMAINS:
        clean = images_by_index(clean_root, domain)
        nightshade = images_by_index(nightshade_root, domain)
        missing = sorted(set(nightshade) - set(clean))
        extra = sorted(set(clean) - set(nightshade))
        if missing or extra:
            raise ValueError(
                f"Index mismatch for {domain}: "
                f"{len(missing)} missing clean, {len(extra)} missing Nightshade"
            )
        for data_index in sorted(clean):
            pairs.append(
                Pair(
                    domain=domain,
                    data_index=data_index,
                    clean_path=clean[data_index],
                    nightshade_path=nightshade[data_index],
                    output_path=output_root / domain / f"{domain}_{data_index}.png",
                )
            )
    return pairs


def load_targets(
    args: argparse.Namespace,
    spec: TargetSpec,
) -> dict[tuple[str, str], float]:
    if args.target_source == "compute":
        return {}
    if not PERTURBATION_CSV.is_file():
        if args.target_source == "csv":
            raise FileNotFoundError(f"Missing perturbation CSV: {PERTURBATION_CSV}")
        return {}

    data = pd.read_csv(
        PERTURBATION_CSV,
        dtype={
            "domain": "string",
            "data_index": "string",
            "poisoning": "string",
            "poisoning_factor": "string",
        },
    )
    selected = data.loc[
        data["poisoning"].eq(spec.poisoning)
        & data["poisoning_factor"].eq(spec.factor),
        ["domain", "data_index", "lpips"],
    ].copy()
    targets: dict[tuple[str, str], float] = {}
    for row in selected.itertuples(index=False):
        targets[(str(row.domain), f"{int(row.data_index):04d}")] = float(row.lpips)
    if args.target_source == "csv" and not targets:
        raise ValueError(
            "No matching Nightshade LPIPS rows in perturbation CSV for "
            f"{spec.poisoning}/{spec.factor}"
        )
    return targets


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(DEVICE)


def lpips_input(batch: torch.Tensor) -> torch.Tensor:
    return batch.mul(2.0).sub(1.0)


def stable_seed(base_seed: int, domain: str, data_index: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{domain}:{data_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def candidate_lpips(
    model: torch.nn.Module,
    clean: torch.Tensor,
    direction: torch.Tensor,
    alphas: np.ndarray,
) -> np.ndarray:
    alpha_tensor = torch.from_numpy(alphas.astype(np.float32)).to(DEVICE).view(-1, 1, 1, 1)
    clean_batch = clean.unsqueeze(0).expand(len(alphas), -1, -1, -1)
    candidates = torch.clamp(clean_batch + alpha_tensor * direction.unsqueeze(0), 0.0, 1.0)
    with torch.inference_mode():
        values = model(lpips_input(clean_batch), lpips_input(candidates))
    return values.flatten().detach().cpu().numpy().astype(np.float64)


def matched_alpha(
    model: torch.nn.Module,
    clean: torch.Tensor,
    direction: torch.Tensor,
    target: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    if target <= 0:
        return 0.0, 0.0

    coarse = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.geomspace(1e-5, args.max_alpha, args.coarse_steps),
        ]
    )
    values = candidate_lpips(model, clean, direction, coarse)
    best = int(np.argmin(np.abs(values - target)))
    best_alpha = float(coarse[best])
    best_value = float(values[best])

    lo_index = max(0, best - 1)
    hi_index = min(len(coarse) - 1, best + 1)
    lo = float(coarse[lo_index])
    hi = float(coarse[hi_index])

    for _ in range(args.refine_rounds):
        if abs(best_value - target) <= args.tolerance:
            break
        grid = np.linspace(lo, hi, args.refine_steps, dtype=np.float64)
        values = candidate_lpips(model, clean, direction, grid)
        best = int(np.argmin(np.abs(values - target)))
        best_alpha = float(grid[best])
        best_value = float(values[best])
        lo = float(grid[max(0, best - 1)])
        hi = float(grid[min(len(grid) - 1, best + 1)])

    return best_alpha, best_value


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.cpu().numpy(), mode="RGB").save(path, format="PNG")


def process_target(
    args: argparse.Namespace,
    spec: TargetSpec,
    model: torch.nn.Module,
) -> None:
    clean_root = DATA_ROOT / args.clean_dataset
    nightshade_root = DATA_ROOT / spec.nightshade_dataset
    output_root = DATA_ROOT / spec.output_dataset
    pairs = collect_pairs(clean_root, nightshade_root, output_root)
    csv_targets = load_targets(args, spec)

    rows: list[dict] = []
    for count, pair in enumerate(pairs, start=1):
        if pair.output_path.is_file() and not args.overwrite:
            continue

        clean = load_image(pair.clean_path)
        target = csv_targets.get((pair.domain, pair.data_index))
        if target is None:
            nightshade = load_image(pair.nightshade_path)
            target = float(
                candidate_lpips(
                    model,
                    clean,
                    nightshade - clean,
                    np.array([1.0], dtype=np.float64),
                )[0]
            )

        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(stable_seed(args.seed, pair.domain, pair.data_index))
        direction = torch.randn(clean.shape, generator=generator, device=DEVICE)
        alpha, matched = matched_alpha(model, clean, direction, target, args)
        output = torch.clamp(clean + alpha * direction, 0.0, 1.0)
        save_image(output, pair.output_path)
        rows.append(
            {
                "domain": pair.domain,
                "data_index": pair.data_index,
                "target_poisoning": spec.poisoning,
                "target_poisoning_factor": spec.factor,
                "target_lpips": target,
                "matched_lpips": matched,
                "alpha": alpha,
                "abs_error": abs(matched - target),
                "output_path": store_path(pair.output_path),
            }
        )

        if count % 100 == 0 or count == len(pairs):
            print(f"[{count}/{len(pairs)}] {pair.domain}/{pair.data_index}")

    report = pd.DataFrame(rows)
    report_path = output_root / "lpips_matching_report.csv"
    if not report.empty:
        report.to_csv(report_path, index=False, float_format="%.10g")
    print(f"Target: {spec.poisoning}/{spec.factor}")
    print(f"Pairs: {len(pairs)}")
    print(f"Written: {len(rows)}")
    print(f"Output dataset: {output_root}")
    print(f"Report: {report_path}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for LPIPS matching.")

    specs = parse_target_specs(args)
    model = lpips.LPIPS(net="alex").to(DEVICE).eval()
    for spec in specs:
        process_target(args, spec, model)


if __name__ == "__main__":
    main()
