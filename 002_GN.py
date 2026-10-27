#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from path_config import CLEAN_DATASET, DATA_ROOT, GAUSSIAN_DATASET


INPUT_ROOT = DATA_ROOT / CLEAN_DATASET
OUTPUT_SPECS = [
    (GAUSSIAN_DATASET, 0.04),
]

BASE_SEED = 20260518
MAX_WORKERS = 16
MAX_ALPHA = 8.0
BINARY_SEARCH_STEPS = 28

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    files.sort()
    return files


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_seed(relative_path: Path) -> int:
    payload = f"{BASE_SEED}:{relative_path.as_posix()}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def build_base_noise(shape: tuple[int, ...], relative_path: Path) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(relative_path))
    return rng.standard_normal(shape).astype(np.float32)


def render_noisy_uint8(
    clean_uint8: np.ndarray,
    base_noise: np.ndarray,
    alpha: float,
) -> np.ndarray:
    clean_float = clean_uint8.astype(np.float32) / 255.0
    noisy_float = np.clip(clean_float + alpha * base_noise, 0.0, 1.0)
    return np.rint(noisy_float * 255.0).astype(np.uint8)


def attained_linf(clean_uint8: np.ndarray, noisy_uint8: np.ndarray) -> float:
    diff = np.abs(
        noisy_uint8.astype(np.int16) - clean_uint8.astype(np.int16)
    )
    return float(diff.max()) / 255.0


def find_best_alpha(
    clean_uint8: np.ndarray,
    base_noise: np.ndarray,
    epsilon: float,
) -> tuple[float, np.ndarray, float]:
    lo = 0.0
    hi = max(epsilon, 1e-6)
    hi_image = render_noisy_uint8(clean_uint8, base_noise, hi)
    hi_linf = attained_linf(clean_uint8, hi_image)

    while hi_linf <= epsilon and hi < MAX_ALPHA:
        lo = hi
        hi *= 2.0
        hi_image = render_noisy_uint8(clean_uint8, base_noise, hi)
        hi_linf = attained_linf(clean_uint8, hi_image)

    if hi > MAX_ALPHA:
        hi = MAX_ALPHA
        hi_image = render_noisy_uint8(clean_uint8, base_noise, hi)
        hi_linf = attained_linf(clean_uint8, hi_image)

    best_alpha = lo
    best_image = render_noisy_uint8(clean_uint8, base_noise, best_alpha)
    best_linf = attained_linf(clean_uint8, best_image)

    for _ in range(BINARY_SEARCH_STEPS):
        mid = (lo + hi) / 2.0
        mid_image = render_noisy_uint8(clean_uint8, base_noise, mid)
        mid_linf = attained_linf(clean_uint8, mid_image)

        if mid_linf <= epsilon:
            lo = mid
            best_alpha = mid
            best_image = mid_image
            best_linf = mid_linf
        else:
            hi = mid

    return best_alpha, best_image, best_linf


def split_color_and_aux(array: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if array.ndim == 2:
        return array, None
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3], array[:, :, 3:]
    if array.ndim == 3 and array.shape[2] in (1, 3):
        return array, None
    raise ValueError(f"Unsupported image shape: {array.shape}")


def merge_color_and_aux(color: np.ndarray, aux: np.ndarray | None) -> np.ndarray:
    if aux is None:
        return color
    return np.concatenate([color, aux], axis=2)


def save_image(array: np.ndarray, dst_path: Path) -> None:
    ensure_parent(dst_path)
    Image.fromarray(array).save(dst_path)


def process_one(src_path: Path):
    relative_path = src_path.relative_to(INPUT_ROOT)

    with Image.open(src_path) as image:
        original = np.asarray(image)

    if original.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image: {src_path}")

    color_part, aux_part = split_color_and_aux(original)
    base_noise = build_base_noise(color_part.shape, relative_path)

    results = []
    for output_name, epsilon in OUTPUT_SPECS:
        alpha, noisy_color, actual_linf = find_best_alpha(
            color_part,
            base_noise,
            epsilon,
        )
        noisy_full = merge_color_and_aux(noisy_color, aux_part)
        dst_path = DATA_ROOT / output_name / relative_path
        save_image(noisy_full, dst_path)
        results.append((output_name, alpha, actual_linf))

    return str(relative_path), results


def main() -> None:
    files = collect_images(INPUT_ROOT)
    if not files:
        raise FileNotFoundError(f"No image files found under: {INPUT_ROOT}")

    for output_name, _ in OUTPUT_SPECS:
        (DATA_ROOT / output_name).mkdir(parents=True, exist_ok=True)

    summary = {
        output_name: {
            "alphas": [],
            "linfs": [],
        }
        for output_name, _ in OUTPUT_SPECS
    }

    print(f"INPUT_ROOT  : {INPUT_ROOT}")
    print(f"NUM_FILES   : {len(files)}")
    print(f"MAX_WORKERS : {MAX_WORKERS}")
    for output_name, epsilon in OUTPUT_SPECS:
        print(f"OUTPUT      : {DATA_ROOT / output_name} (linf <= {epsilon:.3f})")

    success = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, src_path) for src_path in files]
        for index, future in enumerate(as_completed(futures), 1):
            rel_path, results = future.result()
            success += 1

            for output_name, alpha, actual_linf in results:
                summary[output_name]["alphas"].append(alpha)
                summary[output_name]["linfs"].append(actual_linf)

            if index % 100 == 0 or index == len(files):
                print(f"[{index}/{len(files)}] processed")

    print("Done.")
    for output_name, epsilon in OUTPUT_SPECS:
        alphas = np.asarray(summary[output_name]["alphas"], dtype=np.float64)
        linfs = np.asarray(summary[output_name]["linfs"], dtype=np.float64)
        print(
            f"{output_name}: "
            f"alpha(min/mean/max)=({alphas.min():.6f}, {alphas.mean():.6f}, {alphas.max():.6f}) | "
            f"linf(min/mean/max)=({linfs.min():.6f}, {linfs.mean():.6f}, {linfs.max():.6f}) | "
            f"target={epsilon:.6f}"
        )


if __name__ == "__main__":
    main()
