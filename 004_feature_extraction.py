from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image

from path_config import (
    CLEAN_DATASET,
    DATA_ROOT,
    DOMAINNET_DATASETS,
    GAUSSIAN_DATASET,
    GAUSSIAN_LPIPS_DATASET,
    NIGHTSHADE_DATASET,
    RESULT_ROOT,
    resolve_stored_path,
    store_path,
)

try:
    import pyarrow  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "pyarrow is required for feature metadata. Install it with: pip install pyarrow"
    ) from exc


TRANSFORMATION_ROOT = DATA_ROOT / "Transformation"
OUTPUT_ROOT = RESULT_ROOT / "features"

DATASET_NAMES = DOMAINNET_DATASETS

DOWNSCALE_FACTORS = (0.75, 0.6, 0.5, 0.4, 0.3, 0.25)
GAUSSIAN_SIGMAS = (0.5, 1.0, 2.0, 3.0)
JPEG_QUALITIES = (95, 85, 75, 50)

KNOWN_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

BATCH_SIZE = 10
OVERWRITE = False
FEATURE_DTYPE = np.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENABLED_EXTRACTORS = (
    "sd_vae",
    "clip_vitl14_336",
    "dinov2_vitb14",
    "resnet50",
    "convnext_tiny",
    "vit_b16_384",
)

FEATURE_DIMS = {
    "sd_vae": 16384,
    "clip_vitl14_336": 768,
    "dinov2_vitb14": 768,
    "resnet50": 2048,
    "convnext_tiny": 768,
    "vit_b16_384": 768,
}

METADATA_COLUMNS = [
    "row_id",
    "image_id",
    "image_path",
    "dataset_name",
    "domain",
    "data_index",
    "poisoning",
    "poisoning_factor",
    "transformation",
    "transformation_factor",
    "image_width",
    "image_height",
]
SORT_COLUMNS = [
    "dataset_name",
    "transformation",
    "transformation_factor",
    "domain",
    "data_index",
    "image_path",
]

SD_VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
SD_VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
SD_VAE_MANIFEST_NAME = "model.json"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
DINO_MODEL_NAME = "dinov2_vitb14"
TIMM_VIT_MODEL_NAME = "vit_base_patch16_384"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

INDEX_RE = re.compile(r"(?P<index>\d+)$")


@dataclass(frozen=True)
class DatasetLocation:
    root: Path
    dataset_name: str
    transformation: str
    transformation_factor: str


@dataclass
class ExtractorRuntime:
    model: torch.nn.Module
    preprocess: Callable[[list[Path]], torch.Tensor]
    extract: Callable[[torch.Tensor], torch.Tensor]


def factor_text(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return str(value)


def dataset_locations() -> list[DatasetLocation]:
    locations: list[DatasetLocation] = []

    for dataset_name in DATASET_NAMES:
        locations.append(
            DatasetLocation(
                root=DATA_ROOT / dataset_name,
                dataset_name=dataset_name,
                transformation="original",
                transformation_factor="none",
            )
        )

    for factor in DOWNSCALE_FACTORS:
        for dataset_name in DATASET_NAMES:
            locations.append(
                DatasetLocation(
                    root=(
                        TRANSFORMATION_ROOT
                        / "downscaling"
                        / f"factor_{factor_text(factor)}"
                        / dataset_name
                    ),
                    dataset_name=dataset_name,
                    transformation="downscaling",
                    transformation_factor=factor_text(factor),
                )
            )

    for sigma in GAUSSIAN_SIGMAS:
        for dataset_name in DATASET_NAMES:
            locations.append(
                DatasetLocation(
                    root=(
                        TRANSFORMATION_ROOT
                        / "gaussian blur"
                        / f"sigma_{sigma:.1f}"
                        / dataset_name
                    ),
                    dataset_name=dataset_name,
                    transformation="gaussian_blur",
                    transformation_factor=f"{sigma:.1f}",
                )
            )

    for quality in JPEG_QUALITIES:
        for dataset_name in DATASET_NAMES:
            locations.append(
                DatasetLocation(
                    root=(
                        TRANSFORMATION_ROOT
                        / "jpeg_comp"
                        / f"quality_{quality}"
                        / dataset_name
                    ),
                    dataset_name=dataset_name,
                    transformation="jpeg_comp",
                    transformation_factor=str(quality),
                )
            )

    missing_roots = [location.root for location in locations if not location.root.is_dir()]
    if missing_roots:
        missing_text = "\n".join(f"  - {path}" for path in missing_roots)
        raise FileNotFoundError(f"Required dataset directories are missing:\n{missing_text}")

    return locations


def parse_poisoning(dataset_name: str) -> tuple[str, str]:
    mapping = {
        CLEAN_DATASET: ("none", "none"),
        NIGHTSHADE_DATASET: ("NS_linf", "0.04"),
        GAUSSIAN_DATASET: ("GN_linf", "0.04"),
        GAUSSIAN_LPIPS_DATASET: ("GN_star_linf", "0.04"),
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return mapping[dataset_name]


def infer_domain_from_filename(image_path: Path) -> str:
    stem = image_path.stem.lower()
    for domain in KNOWN_DOMAINS:
        if stem.startswith(f"{domain}_") or stem.startswith(f"noisy_{domain}_"):
            return domain
    raise ValueError(f"Could not infer domain from filename: {image_path}")


def infer_data_index(image_path: Path) -> str:
    match = INDEX_RE.search(image_path.stem)
    if match is None:
        raise ValueError(f"Could not infer data index from filename: {image_path}")
    return f"{int(match.group('index')):04d}"


def iter_location_images(location: DatasetLocation):
    domain_directories = sorted(path for path in location.root.iterdir() if path.is_dir())

    if domain_directories:
        unknown_domains = [
            path for path in domain_directories if path.name not in KNOWN_DOMAINS
        ]
        if unknown_domains:
            unknown_text = "\n".join(f"  - {path}" for path in unknown_domains)
            raise ValueError(f"Unknown domain directories:\n{unknown_text}")

        for domain_directory in domain_directories:
            for image_path in sorted(domain_directory.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                    yield domain_directory.name, image_path
        return

    for image_path in sorted(location.root.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            yield infer_domain_from_filename(image_path), image_path


def discover_images(locations: list[DatasetLocation]) -> list[dict]:
    records: list[dict] = []

    for location in locations:
        poisoning, poisoning_factor = parse_poisoning(location.dataset_name)
        location_count = 0

        for domain, image_path in iter_location_images(location):
            if domain not in KNOWN_DOMAINS:
                raise ValueError(f"Unknown domain '{domain}' for {image_path}")

            data_index = infer_data_index(image_path)
            relative_path = store_path(image_path)
            records.append(
                {
                    "image_id": (
                        f"{location.dataset_name}_{domain}_{data_index}"
                    ),
                    "image_path": relative_path,
                    "dataset_name": location.dataset_name,
                    "domain": domain,
                    "data_index": data_index,
                    "poisoning": poisoning,
                    "poisoning_factor": poisoning_factor,
                    "transformation": location.transformation,
                    "transformation_factor": location.transformation_factor,
                }
            )
            location_count += 1

        if location_count == 0:
            raise FileNotFoundError(f"No images found in required directory: {location.root}")

    image_paths = [record["image_path"] for record in records]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate image paths were discovered in the configured datasets.")

    return records


def build_metadata(discovered_records: list[dict]) -> pd.DataFrame:
    records: list[dict] = []

    for record in discovered_records:
        image_path = resolve_stored_path(record["image_path"])
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        records.append(
            {
                **record,
                "image_width": image_width,
                "image_height": image_height,
            }
        )

    metadata = pd.DataFrame(records)
    metadata = metadata.sort_values(
        SORT_COLUMNS,
        kind="mergesort",
        ignore_index=True,
    )
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    metadata = metadata.loc[:, METADATA_COLUMNS]
    return metadata


def validate_metadata_columns(metadata: pd.DataFrame) -> None:
    missing_columns = [
        column for column in METADATA_COLUMNS if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")


def validate_existing_metadata(
    metadata: pd.DataFrame,
    discovered_records: list[dict],
) -> None:
    validate_metadata_columns(metadata)

    if len(metadata) != len(discovered_records):
        raise ValueError(
            "Existing metadata row count does not match discovered images: "
            f"{len(metadata)} != {len(discovered_records)}"
        )

    expected_row_ids = np.arange(len(metadata), dtype=np.int64)
    actual_row_ids = metadata["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_row_ids, expected_row_ids):
        raise ValueError("Existing metadata row_id must be sequential from 0 to N-1.")

    sorted_indices = metadata.sort_values(
        SORT_COLUMNS,
        kind="mergesort",
    ).index.to_numpy()
    if not np.array_equal(sorted_indices, expected_row_ids):
        raise ValueError(
            "Existing metadata does not follow the required row sorting order."
        )

    discovered_paths = {record["image_path"] for record in discovered_records}
    metadata_paths = set(metadata["image_path"].astype(str))
    if metadata_paths != discovered_paths:
        missing = sorted(discovered_paths - metadata_paths)
        extra = sorted(metadata_paths - discovered_paths)
        raise ValueError(
            "Existing metadata image paths do not match the configured datasets. "
            f"Missing: {missing[:10]}; extra: {extra[:10]}"
        )

    missing_files = [
        str(resolve_stored_path(path))
        for path in metadata["image_path"].astype(str)
        if not resolve_stored_path(path).is_file()
    ]
    if missing_files:
        missing_text = "\n".join(f"  - {path}" for path in missing_files[:100])
        raise FileNotFoundError(f"Metadata image paths are missing:\n{missing_text}")


def append_metadata(
    metadata: pd.DataFrame,
    discovered_records: list[dict],
) -> pd.DataFrame:
    validate_metadata_columns(metadata)

    expected_row_ids = np.arange(len(metadata), dtype=np.int64)
    actual_row_ids = metadata["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_row_ids, expected_row_ids):
        raise ValueError("Existing metadata row_id must be sequential from 0 to N-1.")

    metadata_paths = set(metadata["image_path"].astype(str))
    discovered_paths = {record["image_path"] for record in discovered_records}
    removed_paths = sorted(metadata_paths - discovered_paths)
    if removed_paths:
        raise ValueError(
            "Configured datasets no longer contain existing metadata image paths. "
            f"Examples: {removed_paths[:10]}"
        )

    new_records = [
        record
        for record in discovered_records
        if record["image_path"] not in metadata_paths
    ]
    if not new_records:
        print("Using existing metadata: no new rows discovered")
        return metadata

    new_metadata = build_metadata(new_records)
    new_metadata = new_metadata.drop(columns=["row_id"])
    new_metadata.insert(
        0,
        "row_id",
        np.arange(
            len(metadata),
            len(metadata) + len(new_metadata),
            dtype=np.int64,
        ),
    )
    combined = pd.concat(
        [metadata.loc[:, METADATA_COLUMNS], new_metadata.loc[:, METADATA_COLUMNS]],
        ignore_index=True,
    )
    validate_metadata_columns(combined)
    return combined


def load_or_create_metadata() -> pd.DataFrame:
    locations = dataset_locations()
    discovered_records = discover_images(locations)
    metadata_path = OUTPUT_ROOT / "feature_metadata.parquet"

    if metadata_path.is_file() and not OVERWRITE:
        metadata = pd.read_parquet(metadata_path, engine="pyarrow")
        combined = append_metadata(metadata, discovered_records)
        if len(combined) != len(metadata):
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(metadata_path, engine="pyarrow", index=False)
            print(
                f"Appended metadata rows: {len(combined) - len(metadata)} "
                f"-> {metadata_path}"
            )
        else:
            print(f"Using existing metadata: {metadata_path}")
        return combined

    metadata = build_metadata(discovered_records)
    validate_metadata_columns(metadata)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(metadata_path, engine="pyarrow", index=False)
    print(f"Saved metadata: {metadata_path}")
    return metadata


def metadata_image_paths(metadata: pd.DataFrame) -> list[Path]:
    paths = [resolve_stored_path(path) for path in metadata["image_path"].astype(str)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing[:100])
        raise FileNotFoundError(f"Feature input images are missing:\n{missing_text}")
    return paths


def load_rgb_arrays(image_paths: list[Path], size: int, resize: bool) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if resize:
                rgb = rgb.resize((size, size), resample=Image.Resampling.BICUBIC)
            elif rgb.size != (size, size):
                raise ValueError(
                    f"Expected {size}x{size}, found {rgb.width}x{rgb.height}: {image_path}"
                )
            arrays.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    return np.stack(arrays, axis=0)


def preprocess_minus1_1(image_paths: list[Path], size: int) -> torch.Tensor:
    arrays = load_rgb_arrays(image_paths, size=size, resize=False)
    batch = torch.from_numpy(arrays).permute(0, 3, 1, 2).contiguous()
    return batch.mul_(2.0).sub_(1.0)


def preprocess_normalized(
    image_paths: list[Path],
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    resize: bool,
) -> torch.Tensor:
    arrays = load_rgb_arrays(image_paths, size=size, resize=resize)
    batch = torch.from_numpy(arrays).permute(0, 3, 1, 2).contiguous()
    mean_tensor = torch.tensor(mean, dtype=batch.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=batch.dtype).view(1, 3, 1, 1)
    return batch.sub_(mean_tensor).div_(std_tensor)


def load_sd_vae_model() -> torch.nn.Module:
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise ImportError(
            "diffusers is required for sd_vae. Install it with: pip install diffusers"
        ) from exc

    model = AutoencoderKL.from_pretrained(
        SD_VAE_MODEL_ID,
        revision=SD_VAE_REVISION,
    )
    print(f"Loaded SD VAE: {SD_VAE_MODEL_ID}@{SD_VAE_REVISION}")
    return model.to(DEVICE).eval()


def load_extractor(extractor_name: str) -> ExtractorRuntime:
    if extractor_name == "sd_vae":
        model = load_sd_vae_model()

        def preprocess(paths: list[Path]) -> torch.Tensor:
            return preprocess_minus1_1(paths, size=512)

        def extract(batch: torch.Tensor) -> torch.Tensor:
            return model.encode(batch).latent_dist.mode().flatten(1)

        return ExtractorRuntime(model=model, preprocess=preprocess, extract=extract)

    if extractor_name == "clip_vitl14_336":
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "transformers is required for clip_vitl14_336. "
                "Install it with: pip install transformers"
            ) from exc

        model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
        model = model.to(DEVICE).eval()

        def preprocess(paths: list[Path]) -> torch.Tensor:
            return preprocess_normalized(
                paths,
                size=336,
                mean=CLIP_MEAN,
                std=CLIP_STD,
                resize=True,
            )

        def extract(batch: torch.Tensor) -> torch.Tensor:
            return model(pixel_values=batch).image_embeds

        return ExtractorRuntime(model=model, preprocess=preprocess, extract=extract)

    if extractor_name == "dinov2_vitb14":
        model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL_NAME)
        model = model.to(DEVICE).eval()

        def preprocess(paths: list[Path]) -> torch.Tensor:
            return preprocess_normalized(
                paths,
                size=518,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                resize=True,
            )

        def extract(batch: torch.Tensor) -> torch.Tensor:
            # dinov2_vitb14 forward returns the CLS token feature.
            return model(batch)

        return ExtractorRuntime(model=model, preprocess=preprocess, extract=extract)

    if extractor_name == "resnet50":
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for resnet50. Install it with: pip install torchvision"
            ) from exc

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        model = model.to(DEVICE).eval()

        def preprocess(paths: list[Path]) -> torch.Tensor:
            return preprocess_normalized(
                paths,
                size=512,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                resize=False,
            )

        def extract(batch: torch.Tensor) -> torch.Tensor:
            return model(batch)

        return ExtractorRuntime(model=model, preprocess=preprocess, extract=extract)

    if extractor_name == "convnext_tiny":
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for convnext_tiny. "
                "Install it with: pip install torchvision"
            ) from exc

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model = model.to(DEVICE).eval()

        def preprocess(paths: list[Path]) -> torch.Tensor:
            return preprocess_normalized(
                paths,
                size=512,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                resize=False,
            )

        def extract(batch: torch.Tensor) -> torch.Tensor:
            feature_map = model.features(batch)
            pooled = model.avgpool(feature_map)
            return torch.flatten(pooled, 1)

        return ExtractorRuntime(model=model, preprocess=preprocess, extract=extract)

    if extractor_name == "vit_b16_384":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for vit_b16_384. Install it with: pip install timm"
            ) from exc

        model = timm.create_model(
            TIMM_VIT_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )
        data_config = timm.data.resolve_model_data_config(model)
        mean = tuple(data_config.get("mean", IMAGENET_MEAN))
        std = tuple(data_config.get("std", IMAGENET_STD))
        model = model.to(DEVICE).eval()

        def preprocess(paths: list[Path]) -> torch.Tensor:
            return preprocess_normalized(
                paths,
                size=384,
                mean=mean,
                std=std,
                resize=True,
            )

        def extract(batch: torch.Tensor) -> torch.Tensor:
            return model(batch)

        return ExtractorRuntime(model=model, preprocess=preprocess, extract=extract)

    raise ValueError(f"Unknown enabled extractor: {extractor_name}")


def validate_existing_feature(
    output_path: Path,
    expected_dim: int,
) -> np.ndarray:
    features = np.load(output_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Existing feature shape mismatch for {output_path}: "
            f"expected (*, {expected_dim}), found {features.shape}"
        )
    if features.dtype != FEATURE_DTYPE:
        raise ValueError(
            f"Existing feature dtype mismatch for {output_path}: "
            f"expected {FEATURE_DTYPE}, found {features.dtype}"
        )
    return features


def copy_existing_rows(
    source: np.ndarray,
    destination: np.ndarray,
    rows: int,
) -> None:
    for start in range(0, rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, rows)
        destination[start:end] = source[start:end]


def sd_vae_manifest_matches(output_path: Path) -> bool:
    manifest_path = output_path.parent / SD_VAE_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        manifest.get("model_id") == SD_VAE_MODEL_ID
        and manifest.get("revision") == SD_VAE_REVISION
        and manifest.get("feature_extraction") == "latent_dist.mode().flatten(1)"
        and manifest.get("feature_dtype") == str(np.dtype(FEATURE_DTYPE))
        and manifest.get("feature_dim") == FEATURE_DIMS["sd_vae"]
    )


def save_sd_vae_manifest(output_path: Path) -> None:
    manifest = {
        "model_id": SD_VAE_MODEL_ID,
        "revision": SD_VAE_REVISION,
        "feature_extraction": "latent_dist.mode().flatten(1)",
        "input_size": 512,
        "input_range": "[-1, 1]",
        "feature_dtype": str(np.dtype(FEATURE_DTYPE)),
        "feature_dim": FEATURE_DIMS["sd_vae"],
    }
    manifest_path = output_path.parent / SD_VAE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def extract_encoder_features(
    extractor_name: str,
    metadata: pd.DataFrame,
    image_paths: list[Path],
) -> None:
    feature_dim = FEATURE_DIMS[extractor_name]
    output_path = OUTPUT_ROOT / extractor_name / "features.float16.npy"
    start_row = 0
    existing_features: np.ndarray | None = None
    extractor_overwrite = OVERWRITE
    if (
        extractor_name == "sd_vae"
        and output_path.is_file()
        and not sd_vae_manifest_matches(output_path)
    ):
        extractor_overwrite = True
        print(
            "Recalculating sd_vae because its model manifest is missing "
            "or does not match the pinned checkpoint"
        )

    if output_path.is_file() and not extractor_overwrite:
        existing_features = validate_existing_feature(output_path, feature_dim)
        if existing_features.shape[0] > len(metadata):
            raise ValueError(
                f"Existing feature rows exceed metadata rows for {extractor_name}: "
                f"{existing_features.shape[0]} > {len(metadata)}"
            )
        if existing_features.shape[0] == len(metadata):
            print(f"Skipping existing encoder: {extractor_name} ({output_path})")
            return
        start_row = existing_features.shape[0]
        print(
            f"Appending encoder rows: {extractor_name} "
            f"{start_row}->{len(metadata)}"
        )

    runtime = load_extractor(extractor_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".tmp.npy",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    features = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=FEATURE_DTYPE,
        shape=(len(metadata), feature_dim),
    )
    if existing_features is not None:
        copy_existing_rows(existing_features, features, start_row)

    with torch.inference_mode():
        for start in range(start_row, len(metadata), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(metadata))
            batch = runtime.preprocess(image_paths[start:end]).to(
                DEVICE,
                non_blocking=True,
            )
            batch_features = runtime.extract(batch)

            if batch_features.ndim != 2 or batch_features.shape != (
                end - start,
                feature_dim,
            ):
                raise RuntimeError(
                    f"Unexpected {extractor_name} batch feature shape: "
                    f"expected {(end - start, feature_dim)}, "
                    f"found {tuple(batch_features.shape)}"
                )

            features[start:end] = (
                batch_features.detach().float().cpu().numpy().astype(FEATURE_DTYPE)
            )

            if end % 500 < BATCH_SIZE or end == len(metadata):
                print(f"[{extractor_name}] {end}/{len(metadata)} rows")

    features.flush()
    del features
    if existing_features is not None:
        del existing_features
    os.replace(temp_path, output_path)
    if extractor_name == "sd_vae":
        save_sd_vae_manifest(output_path)

    saved_features = np.load(output_path, mmap_mode="r", allow_pickle=False)
    assert len(metadata) == saved_features.shape[0]
    assert saved_features.ndim == 2
    assert saved_features.dtype == FEATURE_DTYPE

    print(f"Encoder: {extractor_name}")
    print(f"  rows: {saved_features.shape[0]}")
    print(f"  feature dimension: {saved_features.shape[1]}")
    print(f"  dtype: {saved_features.dtype}")
    print(f"  output: {output_path}")

    del saved_features, runtime
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")

    unknown_extractors = [
        name for name in ENABLED_EXTRACTORS if name not in FEATURE_DIMS
    ]
    if unknown_extractors:
        raise ValueError(f"Unknown enabled extractors: {unknown_extractors}")

    metadata = load_or_create_metadata()
    image_paths = metadata_image_paths(metadata)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Enabled extractors: {ENABLED_EXTRACTORS}")

    for extractor_name in ENABLED_EXTRACTORS:
        extract_encoder_features(extractor_name, metadata, image_paths)


if __name__ == "__main__":
    main()
