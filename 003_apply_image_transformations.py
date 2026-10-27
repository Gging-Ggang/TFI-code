from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("TFI_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_ROOT = DATA_ROOT / "Transformation"

DATASET_NAMES = (
    "D_domainnet_clean",
    "D_domainnet_nightshade_linf_004",
    "D_domainnet_gaussian_linf_004",
    "D_domainnet_gaussian_lpips_matched_ns_linf_004",
)

DOWNSCALE_FACTORS = (0.75, 0.6, 0.5, 0.4, 0.3, 0.25)
GAUSSIAN_SIGMAS = (0.5, 1.0, 2.0, 3.0)
JPEG_QUALITIES = (95, 85, 75, 50)

IMAGE_SIZE = 512
BATCH_SIZE = 256
SAVE_WORKERS = 32
OVERWRITE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {".png"}


def dataset_directories() -> list[Path]:
    """Return only the explicitly configured source datasets."""
    directories = [DATA_ROOT / name for name in DATASET_NAMES]
    missing = [path for path in directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Required dataset directories are missing: {missing}")
    return directories


def image_paths(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def batched(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_batch(paths: list[Path]) -> tuple[np.ndarray, torch.Tensor]:
    arrays: list[np.ndarray] = []

    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} image, "
                    f"but found {rgb.width}x{rgb.height}: {path}"
                )
            arrays.append(np.asarray(rgb, dtype=np.uint8))

    batch_uint8 = np.stack(arrays, axis=0)
    batch = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).contiguous()

    if DEVICE.type == "cuda":
        batch = batch.pin_memory()

    batch = batch.to(DEVICE, non_blocking=True).float().div_(255.0)
    return batch_uint8, batch


def output_path(
    transformation_directory: Path,
    dataset_root: Path,
    source_path: Path,
) -> Path:
    relative_path = source_path.relative_to(dataset_root)
    return transformation_directory / dataset_root.name / relative_path


def tensor_to_uint8_arrays(batch: torch.Tensor) -> np.ndarray:
    return (
        batch.clamp_(0.0, 1.0)
        .mul_(255.0)
        .round_()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def save_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(destination, format="PNG")


def save_batch(
    arrays: np.ndarray,
    source_paths: list[Path],
    dataset_root: Path,
    transformation_directory: Path,
    executor: ThreadPoolExecutor,
) -> int:
    jobs = []
    for array, source_path in zip(arrays, source_paths):
        destination = output_path(
            transformation_directory, dataset_root, source_path
        )
        if destination.exists() and not OVERWRITE:
            continue
        jobs.append(executor.submit(save_png, array, destination))

    for job in jobs:
        job.result()
    return len(jobs)


@torch.no_grad()
def downscale_upscale(batch: torch.Tensor, factor: float) -> torch.Tensor:
    scaled_size = max(1, int(IMAGE_SIZE * factor))
    downscaled = F.interpolate(
        batch,
        size=(scaled_size, scaled_size),
        mode="bilinear",
        align_corners=False,
    )
    return F.interpolate(
        downscaled,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )


def gaussian_kernel(sigma: float, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    radius = math.ceil(3.0 * sigma)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=DEVICE,
        dtype=dtype,
    )
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, *kernel_2d.shape).repeat(3, 1, 1, 1)
    return kernel, radius


@torch.no_grad()
def gaussian_blur(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel, radius = gaussian_kernel(sigma, batch.dtype)
    padded = F.pad(batch, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=3)


def jpeg_compress_to_png(
    array: np.ndarray,
    destination: Path,
    quality: int,
) -> None:
    source_image = Image.fromarray(array, mode="RGB")
    jpeg_buffer = BytesIO()
    source_image.save(jpeg_buffer, format="JPEG", quality=quality)
    jpeg_buffer.seek(0)

    with Image.open(jpeg_buffer) as compressed_image:
        decoded = np.asarray(compressed_image.convert("RGB"), dtype=np.uint8)

    save_png(decoded, destination)


def save_jpeg_variants(
    arrays: np.ndarray,
    source_paths: list[Path],
    dataset_root: Path,
    executor: ThreadPoolExecutor,
) -> int:
    jobs = []
    for array, source_path in zip(arrays, source_paths):
        for quality in JPEG_QUALITIES:
            transformation_directory = (
                OUTPUT_ROOT / "jpeg_comp" / f"quality_{quality}"
            )
            destination = output_path(
                transformation_directory, dataset_root, source_path
            )
            if destination.exists() and not OVERWRITE:
                continue
            jobs.append(
                executor.submit(
                    jpeg_compress_to_png,
                    array,
                    destination,
                    quality,
                )
            )

    for job in jobs:
        job.result()
    return len(jobs)


def expected_output_paths(dataset_root: Path, source_path: Path) -> list[Path]:
    paths: list[Path] = []
    for factor in DOWNSCALE_FACTORS:
        paths.append(
            output_path(
                OUTPUT_ROOT / "downscaling" / f"factor_{factor:g}",
                dataset_root,
                source_path,
            )
        )
    for sigma in GAUSSIAN_SIGMAS:
        paths.append(
            output_path(
                OUTPUT_ROOT / "gaussian blur" / f"sigma_{sigma:.1f}",
                dataset_root,
                source_path,
            )
        )
    for quality in JPEG_QUALITIES:
        paths.append(
            output_path(
                OUTPUT_ROOT / "jpeg_comp" / f"quality_{quality}",
                dataset_root,
                source_path,
            )
        )
    return paths


def missing_source_paths(dataset_root: Path, paths: list[Path]) -> list[Path]:
    if OVERWRITE:
        return paths
    return [
        source_path
        for source_path in paths
        if any(
            not destination.is_file()
            for destination in expected_output_paths(dataset_root, source_path)
        )
    ]


def process_dataset(dataset_root: Path, executor: ThreadPoolExecutor) -> int:
    paths = image_paths(dataset_root)
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {dataset_root}")
    paths_to_process = missing_source_paths(dataset_root, paths)
    if not paths_to_process:
        print(f"[SKIP] {dataset_root.name}: all transformed images already exist")
        return 0

    written = 0
    processed = 0

    for source_paths in batched(paths_to_process, BATCH_SIZE):
        batch_uint8, batch = load_batch(source_paths)

        for factor in DOWNSCALE_FACTORS:
            transformed = downscale_upscale(batch, factor)
            arrays = tensor_to_uint8_arrays(transformed)
            transformation_directory = (
                OUTPUT_ROOT / "downscaling" / f"factor_{factor:g}"
            )
            written += save_batch(
                arrays,
                source_paths,
                dataset_root,
                transformation_directory,
                executor,
            )
            del transformed, arrays

        for sigma in GAUSSIAN_SIGMAS:
            transformed = gaussian_blur(batch, sigma)
            arrays = tensor_to_uint8_arrays(transformed)
            transformation_directory = (
                OUTPUT_ROOT / "gaussian blur" / f"sigma_{sigma:.1f}"
            )
            written += save_batch(
                arrays,
                source_paths,
                dataset_root,
                transformation_directory,
                executor,
            )
            del transformed, arrays

        written += save_jpeg_variants(
            batch_uint8,
            source_paths,
            dataset_root,
            executor,
        )

        processed += len(source_paths)
        print(
            f"[{dataset_root.name}] "
            f"{processed}/{len(paths_to_process)} missing source images processed"
        )

    print(f"[DONE] {dataset_root.name}: {written} transformed images written")
    return written


def main() -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this script.")

    torch.backends.cudnn.benchmark = True
    datasets = dataset_directories()

    print(f"Device: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Output root: {OUTPUT_ROOT}")

    total_written = 0
    with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as executor:
        for dataset_root in datasets:
            total_written += process_dataset(dataset_root, executor)

    print(f"Total transformed images written: {total_written}")


if __name__ == "__main__":
    main()
