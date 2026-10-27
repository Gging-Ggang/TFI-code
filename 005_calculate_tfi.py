from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Hardcoded configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "result" / "features"
METADATA_PATH = FEATURE_ROOT / "feature_metadata.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "result" / "tfi"
OUTPUT_CSV = OUTPUT_ROOT / "tfi.csv"
OUTPUT_NPY = OUTPUT_ROOT / "tfi.float32.npy"

FEATURE_EXTRACTORS = (
    "sd_vae",
    "clip_vitl14_336",
    "dinov2_vitb14",
    "resnet50",
    "convnext_tiny",
    "vit_b16_384",
)
OVERWRITE_OUTPUT = True
FEATURE_DIMS = {
    "sd_vae": 16384,
    "clip_vitl14_336": 768,
    "dinov2_vitb14": 768,
    "resnet50": 2048,
    "convnext_tiny": 768,
    "vit_b16_384": 768,
}

TRANSFORMATIONS = (
    ("downscaling", "0.75", "ld_downscaling_0.75"),
    ("downscaling", "0.6", "ld_downscaling_0.6"),
    ("downscaling", "0.5", "ld_downscaling_0.5"),
    ("downscaling", "0.4", "ld_downscaling_0.4"),
    ("downscaling", "0.3", "ld_downscaling_0.3"),
    ("downscaling", "0.25", "ld_downscaling_0.25"),
    ("gaussian_blur", "0.5", "ld_gaussian_blur_0.5"),
    ("gaussian_blur", "1.0", "ld_gaussian_blur_1.0"),
    ("gaussian_blur", "2.0", "ld_gaussian_blur_2.0"),
    ("gaussian_blur", "3.0", "ld_gaussian_blur_3.0"),
    ("jpeg_comp", "95", "ld_jpeg_comp_95"),
    ("jpeg_comp", "85", "ld_jpeg_comp_85"),
    ("jpeg_comp", "75", "ld_jpeg_comp_75"),
    ("jpeg_comp", "50", "ld_jpeg_comp_50"),
)

IDENTITY_COLUMNS = ["dataset_name", "domain", "data_index"]
CSV_METADATA_COLUMNS = [
    "domain",
    "data_index",
    "feature_extractor",
    "poisoning",
    "poisoning_factor",
]
LD_COLUMNS = [column for _, _, column in TRANSFORMATIONS]
ANGLE_COLUMNS = [
    column.replace("ld_", "angle_", 1)
    for column in LD_COLUMNS
]
BATCH_SIZE = 256


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing feature metadata: {METADATA_PATH}")

    metadata = pd.read_parquet(METADATA_PATH)
    required_columns = [
        "row_id",
        "dataset_name",
        "domain",
        "data_index",
        "poisoning",
        "poisoning_factor",
        "transformation",
        "transformation_factor",
    ]
    missing_columns = [
        column for column in required_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Feature metadata is missing columns: {missing_columns}")

    expected_row_ids = np.arange(len(metadata), dtype=np.int64)
    actual_row_ids = metadata["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_row_ids, expected_row_ids):
        raise ValueError("Feature metadata row_id must be sequential from 0 to N-1")

    string_columns = [
        "dataset_name",
        "domain",
        "data_index",
        "poisoning",
        "poisoning_factor",
        "transformation",
        "transformation_factor",
    ]
    for column in string_columns:
        metadata[column] = metadata[column].astype(str)

    return metadata


def original_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    originals = metadata.loc[
        metadata["transformation"].eq("original"),
        [
            "row_id",
            "dataset_name",
            "domain",
            "data_index",
            "poisoning",
            "poisoning_factor",
        ],
    ].copy()
    originals = originals.sort_values(
        ["dataset_name", "domain", "data_index"],
        kind="mergesort",
        ignore_index=True,
    )

    duplicated = originals.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicated.any():
        examples = originals.loc[duplicated, IDENTITY_COLUMNS].head(10)
        raise ValueError(
            "Original rows are not unique by dataset/domain/index:\n"
            + examples.to_string(index=False)
        )
    if originals.empty:
        raise ValueError("No original feature rows were found")

    return originals


def transformation_row_indices(
    metadata: pd.DataFrame,
    originals: pd.DataFrame,
    require_exact: bool = True,
) -> dict[str, np.ndarray]:
    original_keys = pd.MultiIndex.from_frame(originals[IDENTITY_COLUMNS])
    indices_by_column: dict[str, np.ndarray] = {}

    for transformation, factor, output_column in TRANSFORMATIONS:
        selected = metadata.loc[
            metadata["transformation"].eq(transformation)
            & metadata["transformation_factor"].eq(factor),
            ["row_id", *IDENTITY_COLUMNS],
        ].copy()

        duplicated = selected.duplicated(IDENTITY_COLUMNS, keep=False)
        if duplicated.any():
            examples = selected.loc[duplicated, IDENTITY_COLUMNS].head(10)
            raise ValueError(
                f"Duplicate rows for {transformation} {factor}:\n"
                + examples.to_string(index=False)
            )

        selected = selected.set_index(IDENTITY_COLUMNS)
        selected_keys = selected.index
        missing_keys = original_keys.difference(selected_keys)
        extra_keys = selected_keys.difference(original_keys)
        if len(missing_keys) or (require_exact and len(extra_keys)):
            raise ValueError(
                f"Pairing mismatch for {transformation} {factor}: "
                f"{len(missing_keys)} missing and {len(extra_keys)} extra rows"
            )

        indices_by_column[output_column] = selected.loc[
            original_keys,
            "row_id",
        ].to_numpy(dtype=np.int64)

    return indices_by_column


def load_feature_array(extractor_name: str, expected_rows: int) -> np.ndarray:
    feature_path = FEATURE_ROOT / extractor_name / "features.float16.npy"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing feature array: {feature_path}")

    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2:
        raise ValueError(
            f"Feature array must be two-dimensional: {feature_path} {features.shape}"
        )
    if features.shape[0] != expected_rows:
        raise ValueError(
            f"Feature row mismatch for {extractor_name}: "
            f"expected {expected_rows}, found {features.shape[0]}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(
            f"Feature array must use a floating dtype: {feature_path} {features.dtype}"
        )

    return features


def validate_feature_inventory(metadata_rows: int) -> None:
    for extractor_name in FEATURE_EXTRACTORS:
        feature_path = (
            FEATURE_ROOT / extractor_name / "features.float16.npy"
        )
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing feature array: {feature_path}")
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if features.ndim != 2:
            raise ValueError(
                f"Feature array must be two-dimensional: "
                f"{feature_path} {features.shape}"
            )
        if not np.issubdtype(features.dtype, np.floating):
            raise ValueError(
                f"Feature array must use a floating dtype: "
                f"{feature_path} {features.dtype}"
            )
        if features.shape[1] != FEATURE_DIMS[extractor_name]:
            raise ValueError(
                f"Unexpected feature dimension for {extractor_name}: "
                f"expected {FEATURE_DIMS[extractor_name]}, "
                f"found {features.shape[1]}"
            )

        if features.shape[0] != metadata_rows:
            raise ValueError(
                f"Unexpected feature rows for {extractor_name}: "
                f"expected {metadata_rows}, found {features.shape[0]}"
            )
        print(
            f"Feature array: {extractor_name} "
            f"{features.shape} {features.dtype}"
        )


def calculate_metrics(
    features: np.ndarray,
    original_indices: np.ndarray,
    transformed_indices: dict[str, np.ndarray],
    extractor_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.empty(
        (len(original_indices), len(TRANSFORMATIONS)),
        dtype=np.float32,
    )
    angles = np.empty_like(distances)

    for column_index, (_, _, output_column) in enumerate(TRANSFORMATIONS):
        target_indices = transformed_indices[output_column]

        for start in range(0, len(original_indices), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(original_indices))
            original_batch = np.asarray(
                features[original_indices[start:end]],
                dtype=np.float32,
            )
            transformed_batch = np.asarray(
                features[target_indices[start:end]],
                dtype=np.float32,
            )
            difference = original_batch - transformed_batch
            squared_distance = np.einsum(
                "ij,ij->i",
                difference,
                difference,
                dtype=np.float32,
                optimize=True,
            )
            distances[start:end, column_index] = np.sqrt(squared_distance)

            original_squared_norm = np.einsum(
                "ij,ij->i",
                original_batch,
                original_batch,
                dtype=np.float32,
                optimize=True,
            )
            transformed_squared_norm = np.einsum(
                "ij,ij->i",
                transformed_batch,
                transformed_batch,
                dtype=np.float32,
                optimize=True,
            )
            zero_norm = (original_squared_norm <= 0) | (
                transformed_squared_norm <= 0
            )
            if zero_norm.any():
                raise ValueError(
                    f"Zero-norm feature vectors found for {extractor_name} "
                    f"{output_column}: {int(zero_norm.sum())} rows"
                )

            dot_product = np.einsum(
                "ij,ij->i",
                original_batch,
                transformed_batch,
                dtype=np.float32,
                optimize=True,
            )
            denominator = np.sqrt(
                original_squared_norm * transformed_squared_norm
            )
            cosine_similarity = np.clip(dot_product / denominator, -1.0, 1.0)
            angles[start:end, column_index] = np.degrees(
                np.arccos(cosine_similarity)
            )

        print(
            f"[{extractor_name}] {column_index + 1}/{len(TRANSFORMATIONS)} "
            f"{output_column}"
        )

    if not np.all(np.isfinite(distances)):
        raise ValueError(f"Non-finite latent distances found for {extractor_name}")
    if not np.all(np.isfinite(angles)):
        raise ValueError(f"Non-finite latent angles found for {extractor_name}")

    return distances, angles


def result_metadata(
    originals: pd.DataFrame,
    extractor_name: str,
) -> pd.DataFrame:
    result = originals.loc[
        :, ["domain", "data_index", "poisoning", "poisoning_factor"]
    ].copy()
    result.insert(2, "feature_extractor", extractor_name)
    return result.loc[:, CSV_METADATA_COLUMNS]


def save_outputs(
    metadata_frames: list[pd.DataFrame],
    distance_arrays: list[np.ndarray],
    angle_arrays: list[np.ndarray],
    existing_rows: int = 0,
) -> None:
    output_metadata = pd.concat(metadata_frames, ignore_index=True)
    output_distances = np.concatenate(distance_arrays, axis=0).astype(
        np.float32,
        copy=False,
    )
    output_angles = np.concatenate(angle_arrays, axis=0).astype(
        np.float32,
        copy=False,
    )

    if len(output_metadata) != output_distances.shape[0]:
        raise RuntimeError(
            "Output metadata/TFI row mismatch: "
            f"{len(output_metadata)} != {output_distances.shape[0]}"
        )
    if output_distances.shape[1] != len(LD_COLUMNS):
        raise RuntimeError(
            f"Expected {len(LD_COLUMNS)} LD columns, "
            f"found {output_distances.shape[1]}"
        )
    if output_angles.shape != output_distances.shape:
        raise RuntimeError(
            "Output LD/angle shape mismatch: "
            f"{output_distances.shape} != {output_angles.shape}"
        )

    output_dataframe = output_metadata.copy()
    output_dataframe[LD_COLUMNS] = output_distances
    output_dataframe[ANGLE_COLUMNS] = output_angles
    output_metrics = np.concatenate(
        [output_distances, output_angles],
        axis=1,
    ).astype(np.float32, copy=False)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if existing_rows and OUTPUT_CSV.is_file() and OUTPUT_NPY.is_file():
        output_dataframe.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=False,
            index=False,
            float_format="%.8g",
        )
        existing_metrics = np.load(OUTPUT_NPY, mmap_mode="r", allow_pickle=False)
        expected_existing_shape = (existing_rows, output_metrics.shape[1])
        if existing_metrics.shape != expected_existing_shape:
            raise ValueError(
                "Existing TFI NPY shape mismatch: "
                f"{existing_metrics.shape} != {expected_existing_shape}"
            )
        with tempfile.NamedTemporaryFile(
            dir=OUTPUT_ROOT,
            prefix=f".{OUTPUT_NPY.stem}.",
            suffix=".tmp.npy",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        combined = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.float32,
            shape=(existing_rows + output_metrics.shape[0], output_metrics.shape[1]),
        )
        for start in range(0, existing_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, existing_rows)
            combined[start:end] = existing_metrics[start:end]
        combined[existing_rows:] = output_metrics
        combined.flush()
        del combined, existing_metrics
        os.replace(temp_path, OUTPUT_NPY)
    else:
        output_dataframe.to_csv(
            OUTPUT_CSV,
            index=False,
            float_format="%.8g",
        )
        np.save(OUTPUT_NPY, output_metrics, allow_pickle=False)

    print(f"CSV rows: {len(output_dataframe)}")
    print(f"LD shape: {output_distances.shape}")
    print(f"Angle shape: {output_angles.shape}")
    print(f"Combined NPY shape: {output_metrics.shape}")
    print(f"Combined NPY dtype: {output_metrics.dtype}")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved NPY: {OUTPUT_NPY}")


def existing_result_keys() -> tuple[set[tuple[str, str, str, str, str]], int]:
    if not OUTPUT_CSV.is_file() or not OUTPUT_NPY.is_file():
        return set(), 0

    key_columns = [
        "domain",
        "data_index",
        "feature_extractor",
        "poisoning",
        "poisoning_factor",
    ]
    existing = pd.read_csv(
        OUTPUT_CSV,
        usecols=key_columns,
        dtype={column: "string" for column in key_columns},
    )
    metrics = np.load(OUTPUT_NPY, mmap_mode="r", allow_pickle=False)
    expected_columns = len(LD_COLUMNS) + len(ANGLE_COLUMNS)
    if metrics.shape != (len(existing), expected_columns):
        raise ValueError(
            "Existing TFI CSV/NPY mismatch: "
            f"{len(existing)} rows vs {metrics.shape}"
        )
    keys = set(
        map(
            tuple,
            existing.loc[:, key_columns].astype(str).itertuples(index=False, name=None),
        )
    )
    return keys, len(existing)


def main() -> None:
    metadata = load_metadata()
    originals = original_metadata(metadata)
    validate_feature_inventory(len(metadata))
    if OVERWRITE_OUTPUT:
        existing_keys, existing_rows = set(), 0
    else:
        existing_keys, existing_rows = existing_result_keys()

    metadata_frames: list[pd.DataFrame] = []
    distance_arrays: list[np.ndarray] = []
    angle_arrays: list[np.ndarray] = []

    print(f"Feature metadata rows: {len(metadata)}")
    print(f"Original rows per extractor: {len(originals)}")
    print(f"Feature extractors: {FEATURE_EXTRACTORS}")

    for extractor_name in FEATURE_EXTRACTORS:
        candidate_metadata = result_metadata(originals, extractor_name)
        keep_mask = np.array(
            [
                tuple(row) not in existing_keys
                for row in candidate_metadata.astype(str).itertuples(
                    index=False,
                    name=None,
                )
            ],
            dtype=bool,
        )
        if not keep_mask.any():
            print(f"[{extractor_name}] TFI rows already complete")
            continue

        selected_originals = originals.loc[keep_mask].reset_index(drop=True)
        selected_result_metadata = result_metadata(
            selected_originals, extractor_name
        )

        selected_original_indices = selected_originals["row_id"].to_numpy(dtype=np.int64)
        selected_transformed_indices = transformation_row_indices(
            metadata,
            selected_originals,
            require_exact=False,
        )
        features = load_feature_array(extractor_name, expected_rows=len(metadata))
        distances, angles = calculate_metrics(
            features,
            selected_original_indices,
            selected_transformed_indices,
            extractor_name,
        )
        metadata_frames.append(selected_result_metadata)
        distance_arrays.append(distances)
        angle_arrays.append(angles)
        del features

    if not metadata_frames:
        print("TFI outputs already contain all configured rows")
        return

    save_outputs(metadata_frames, distance_arrays, angle_arrays, existing_rows=existing_rows)


if __name__ == "__main__":
    main()
