# scripts/match-prediction/prepare_data.py
# Prepares data for training
# - Filters out outliers
# - Creates train/test split
# - Creates encoders for categorical columns and normalizes numerical columns
# (normalization is important, especially with auxiliary tasks, to avoid tasks losses having different scales)
# The main goal of preparing data like this is: it makes training loop simpler, and more efficient because data preparation was a bottleneck, especially when using many aux tasks.
# And the same data can then easily be used in train.py and train_pro.py.
import os
import glob
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.match_prediction import (
    RAW_AZURE_DIR,
    PREPARED_DATA_DIR,
    CHAMPION_ID_ENCODER_PATH,
    TASK_STATS_PATH,
    PATCH_MAPPING_PATH,
    SAMPLE_COUNTS_PATH,
)
from utils.match_prediction.column_definitions import (
    COLUMNS,
    KNOWN_CATEGORICAL_COLUMNS_NAMES,
    get_patch_from_raw_data,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from typing import List, Dict, Tuple
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import gc
import psutil

# Constants
NUM_RECENT_PATCHES = 52  # Number of most recent patches to use for training
PATCH_MIN_GAMES = 10_000  # Minimum number of games in a patch to be considered standalone(otherwise merged with previous patch)
DEBUG = False
DEBUG_FILE_LIMIT = 100  # Maximum number of files to process in debug mode

# Filter constants
MIN_GOLD_15MIN = 2400  # Minimum gold at 15 minutes
MAX_DEATHS_15MIN = 10  # Maximum deaths at 15 minutes
MIN_CS_15MIN_NORMAL = 25  # Minimum CS at 15 minutes for non-support roles
MIN_LEVEL_15MIN = 4  # Minimum level at 15 minutes


def process_champion_ids(file_path: str) -> set:
    """Process a single file to extract unique champion IDs"""
    df = pd.read_parquet(file_path)
    return set(df["champion_ids"].explode().unique())


def create_champion_id_encoder(data_files: List[str]) -> LabelEncoder:
    """Create encoder for champion_ids using parallel processing"""
    print(f"Creating encoder for champion_ids")

    # Use multiprocessing to process files in parallel
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    with mp.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(process_champion_ids, data_files),
                total=len(data_files),
                desc="Processing champion_ids",
            )
        )

    # Combine all unique values
    unique_values = set()
    for result in results:
        unique_values.update(result)

    unique_values = sorted(unique_values)
    unique_values.append("UNKNOWN")

    encoder = LabelEncoder()
    encoder.fit(unique_values)

    return encoder


def process_task_stats(
    file_path: str,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Process a single file to compute task statistics"""
    df = pd.read_parquet(file_path)

    task_sums = {
        task: 0.0
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    }
    task_sumsq = {
        task: 0.0
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    }

    for task, task_def in TASKS.items():
        if task_def.task_type == TaskType.REGRESSION:
            task_sums[task] = df[task].sum()
            task_sumsq[task] = (df[task] ** 2).sum()

    return task_sums, task_sumsq, len(df)


def compute_task_stats(data_files):
    """Compute task statistics using parallel processing"""
    # Use multiprocessing to process files in parallel
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    with mp.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(process_task_stats, data_files),
                total=len(data_files),
                desc="Computing stats",
            )
        )

    # Combine results
    total_task_sums = {
        task: 0.0
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    }
    total_task_sumsq = {
        task: 0.0
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    }
    total_count = 0

    for task_sums, task_sumsq, count in results:
        for task in total_task_sums:
            total_task_sums[task] += task_sums[task]
            total_task_sumsq[task] += task_sumsq[task]
        total_count += count

    # Compute final statistics
    task_means = {task: total_task_sums[task] / total_count for task in total_task_sums}
    task_stds = {
        task: np.sqrt((total_task_sumsq[task] / total_count) - (task_means[task] ** 2))
        for task in total_task_sums
    }

    return task_means, task_stds


def process_patch_counts(file_path: str) -> Dict[float, int]:
    """Process a single file to compute patch counts"""
    df = pd.read_parquet(file_path)
    raw_patches = get_patch_from_raw_data(df)
    patch_counts = defaultdict(int)
    for patch in raw_patches:
        patch_counts[patch] += 1
    return dict(patch_counts)


def compute_patch_mapping(
    input_files: List[str], next_patch_test: bool = False
) -> Dict[float, int]:
    """
    Analyze all files to create a mapping for patch numbers using parallel processing.
    Takes the NUM_RECENT_PATCHES most recent patches, mapping any patch with <200k games to the previous patch.

    If next_patch_test is True:
        - The 2 most recent patches will be mapped to the same value as the 3rd most recent patch
        - This allows testing how well the model generalizes to future patches

    Returns a dictionary mapping original patch numbers to normalized ones (1-NUM_RECENT_PATCHES).
    """
    print("Computing patch distribution...")

    # Use multiprocessing to process files in parallel
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    with mp.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(process_patch_counts, input_files),
                total=len(input_files),
                desc="Processing patches",
            )
        )

    # Combine results
    patch_counts = defaultdict(int)
    for result in results:
        for patch, count in result.items():
            patch_counts[patch] += count

    # All patches in chronological order
    all_patches = sorted(patch_counts.keys())

    # Get the last NUM_RECENT_PATCHES patches
    recent_patches = all_patches[-NUM_RECENT_PATCHES:]
    if len(recent_patches) < NUM_RECENT_PATCHES and not DEBUG:
        raise ValueError(
            f"Not enough patches in the dataset! Need at least {NUM_RECENT_PATCHES} patches."
        )
    elif len(recent_patches) < NUM_RECENT_PATCHES and DEBUG:
        print(
            f"Warning: Only {len(recent_patches)} patches found in debug mode (normally need {NUM_RECENT_PATCHES})"
        )
        if not recent_patches:  # If no patches found, use all patches
            recent_patches = all_patches
            if not recent_patches:
                raise ValueError("No patches found in the dataset!")

    # Create mapping, handling low-data patches
    patch_mapping = {}
    patch_index = 0
    previous_significant_patch = None

    # Process patches from oldest to newest of the recent patches
    for i, patch in enumerate(recent_patches):
        # Handle next_patch_test logic for the most recent patches
        if next_patch_test and i >= len(recent_patches) - 2:
            # Map the 2 most recent patches to the same value as the 3rd most recent patch
            if len(recent_patches) >= 3 and previous_significant_patch is not None:
                patch_mapping[patch] = patch_mapping[previous_significant_patch]
                continue
            else:
                raise ValueError(
                    "Not enough patches for next-patch testing. Need at least 3 recent patches."
                )

        if (
            patch_counts[patch] > PATCH_MIN_GAMES or DEBUG
        ):  # In debug mode, accept all patches
            # This is a significant patch
            patch_mapping[patch] = patch_index
            previous_significant_patch = patch
            patch_index += 1
        elif previous_significant_patch is not None:
            # Map to previous significant patch
            patch_mapping[patch] = patch_mapping[previous_significant_patch]
        else:
            # First patch has low data
            if DEBUG:
                # In debug mode, accept it anyway
                patch_mapping[patch] = patch_index
                previous_significant_patch = patch
                patch_index += 1
            else:
                raise ValueError(f"First patch {patch} has insufficient data!")

    # Log the mapping for transparency
    print("\nPatch mapping (last NUM_RECENT_PATCHES patches):")
    for patch in sorted(patch_mapping.keys()):
        mapped_value = patch_mapping[patch]
        is_test = next_patch_test and patch in recent_patches[-2:]
        status = " (test set)" if is_test else ""
        print(f"Patch {patch} -> {mapped_value}{status} ({patch_counts[patch]} games)")

    return patch_mapping, dict(patch_counts)


def filter_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter out games with potential griefers/AFKs based on specified rules.
    Uses vectorized operations for better performance.
    """
    filter_counts = {
        "original_count": len(df),
        "gold_filter": 0,
        "deaths_filter": 0,
        "cs_filter_normal": 0,
        "level_filter": 0,
    }

    original_count = len(df)
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    teams = [100, 200]

    # Pre-compute column names for better performance
    gold_cols = [
        f"team_{team}_{role}_totalGold_at_900000" for role in roles for team in teams
    ]
    deaths_cols = [
        f"team_{team}_{role}_deaths_at_900000" for role in roles for team in teams
    ]
    cs_cols = [
        f"team_{team}_{role}_creepScore_at_900000"
        for role in roles
        if role != "UTILITY"
        for team in teams
    ]
    level_cols = [
        f"team_{team}_{role}_level_at_900000" for role in roles for team in teams
    ]

    # 1. Gold filter - vectorized operation
    gold_mask = df[gold_cols].min(axis=1) >= MIN_GOLD_15MIN
    filter_counts["gold_filter"] = (~gold_mask).sum()
    df = df[gold_mask]

    # 2. Deaths filter - vectorized operation
    deaths_mask = df[deaths_cols].max(axis=1) <= MAX_DEATHS_15MIN
    filter_counts["deaths_filter"] = (~deaths_mask).sum()
    df = df[deaths_mask]

    # 3. CS filter - vectorized operation
    cs_mask = df[cs_cols].min(axis=1) >= MIN_CS_15MIN_NORMAL
    filter_counts["cs_filter_normal"] = (~cs_mask).sum()
    df = df[cs_mask]

    # 4. Level filter - vectorized operation
    level_mask = df[level_cols].min(axis=1) >= MIN_LEVEL_15MIN
    filter_counts["level_filter"] = (~level_mask).sum()
    df = df[level_mask]

    # Print summary using tqdm.write
    total_filtered = original_count - len(df)
    if DEBUG:
        summary = [
            f"Filtering summary:",
            f"├─ Gold filter: {filter_counts['gold_filter']:,d} rows",
            f"├─ Deaths filter: {filter_counts['deaths_filter']:,d} rows",
            f"├─ CS filter: {filter_counts['cs_filter_normal']:,d} rows",
            f"├─ Level filter: {filter_counts['level_filter']:,d} rows",
            f"└─ Total: {total_filtered:,d} rows ({total_filtered/original_count*100:.1f}%)",
        ]
        tqdm.write("\n".join(summary))

    return df, filter_counts


def prepare_data_optimized(
    input_files,
    output_dir,
    champion_encoder: LabelEncoder,
    task_means,
    task_stds,
    target_file_size: int = 50000,
    next_patch_test: bool = False,
):
    """Optimized version of prepare_data with better performance"""
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Clean up existing files
    for folder in ["train", "test"]:
        for file_path in glob.glob(os.path.join(output_dir, folder, "*.parquet")):
            os.remove(file_path)

    # Use more efficient data structures for buffers
    train_buffer = []
    test_buffer = []
    train_file_counter = 0
    test_file_counter = 0
    train_count = 0
    test_count = 0

    def write_buffer_optimized(buffer, is_train: bool, file_counter: int):
        if not buffer:
            return file_counter

        file_prefix = "train" if is_train else "test"
        output_path = os.path.join(
            output_dir, file_prefix, f"{file_prefix}_{file_counter:04d}.parquet"
        )

        # Convert buffer to DataFrame more efficiently
        df = pd.DataFrame(buffer)
        # Use optimized parquet settings for faster writes
        df.to_parquet(output_path, index=False, compression="snappy", engine="pyarrow")
        return file_counter + 1

    # Load patch mapping to identify recent patches
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_info = pickle.load(f)

    patch_mapping = patch_info["mapping"]

    # Find the 2 most recent patches if next_patch_test is True
    recent_patches = []
    if next_patch_test:
        all_patches = sorted(patch_mapping.keys())
        recent_patches = all_patches[-2:]
        print(f"Using patches {recent_patches} for test set (future patch evaluation)")

    # Pre-compute task lists for efficiency
    regression_tasks = [
        task
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.REGRESSION
    ]
    binary_tasks = [
        task
        for task, task_def in TASKS.items()
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION
        and not task.startswith("win_prediction_")
    ]

    # Shuffle input files for better data distribution
    rng = np.random.default_rng(42)
    rng.shuffle(input_files)

    # Process files in batches for better memory management
    batch_size = 20  # Process multiple files at once for efficiency
    for batch_start in range(0, len(input_files), batch_size):
        batch_files = input_files[batch_start : batch_start + batch_size]

        # Read and process multiple files together
        batch_dfs = []
        for file_path in batch_files:
            try:
                df = pd.read_parquet(file_path)
                if len(df) > 1:
                    batch_dfs.append((df, file_path))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if not batch_dfs:
            continue

        # Process the batch
        # TODO: make intra batch progress updates better, right now it updates only when finished
        for old_df, file_path in tqdm(
            batch_dfs,
            desc=f"Processing batch {batch_start//batch_size + 1}/{(len(input_files) + batch_size - 1)//batch_size}",
        ):
            # Pre-allocate new DataFrame with correct dtypes
            num_rows = len(old_df)

            # Create new DataFrame more efficiently using dict comprehension
            new_data = {}

            # Copy categorical columns directly with proper dtype
            for col in KNOWN_CATEGORICAL_COLUMNS_NAMES:
                new_data[col] = old_df[col].astype("int32", copy=False)

            # Handle patch column
            new_data["patch"] = old_df["patch"].astype("int32", copy=False)

            # Process champion_ids more efficiently using vectorized operations
            champion_ids_transformed = old_df["champion_ids"].apply(
                lambda x: champion_encoder.transform(x)
            )
            new_data["champion_ids"] = champion_ids_transformed

            # Vectorized processing of regression tasks
            for task in regression_tasks:
                if task in old_df.columns:
                    if task_stds[task] != 0:
                        new_data[task] = (
                            (old_df[task] - task_means[task]) / task_stds[task]
                        ).astype("float32", copy=False)
                    else:
                        new_data[task] = (old_df[task] - task_means[task]).astype(
                            "float32", copy=False
                        )

            # Vectorized processing of binary classification tasks
            for task in binary_tasks:
                if task == "win_prediction":
                    new_data[task] = old_df["team_100_win"].astype(
                        "float32", copy=False
                    )
                elif task in old_df.columns:
                    new_data[task] = old_df[task].astype("float32", copy=False)

            # Create DataFrame from dict (more efficient than pre-allocation)
            new_df = pd.DataFrame(new_data)

            # Handle train/test split more efficiently
            if next_patch_test:
                raw_patches = get_patch_from_raw_data(old_df)
                if any(patch in recent_patches for patch in raw_patches.unique()):
                    is_recent_patch = raw_patches.isin(recent_patches)

                    if is_recent_patch.any():
                        test_rows = new_df[is_recent_patch]
                        test_buffer.extend(test_rows.to_dict("records"))
                        test_count += len(test_rows)

                    if (~is_recent_patch).any():
                        train_rows = new_df[~is_recent_patch]
                        train_buffer.extend(train_rows.to_dict("records"))
                        train_count += len(train_rows)
                else:
                    train_buffer.extend(new_df.to_dict("records"))
                    train_count += len(new_df)
            else:
                # More efficient train/test split
                if len(new_df) < 10:
                    train_data = new_df.iloc[:-1].to_dict("records")
                    test_data = new_df.iloc[-1:].to_dict("records")
                else:
                    # Use numpy for faster splitting
                    indices = np.arange(len(new_df))
                    rng.shuffle(indices)
                    split_idx = int(len(new_df) * 0.9)

                    train_indices = indices[:split_idx]
                    test_indices = indices[split_idx:]

                    train_data = new_df.iloc[train_indices].to_dict("records")
                    test_data = new_df.iloc[test_indices].to_dict("records")

                train_buffer.extend(train_data)
                test_buffer.extend(test_data)
                train_count += len(train_data)
                test_count += len(test_data)

            # Write buffers if they exceed target size
            if len(train_buffer) >= target_file_size:
                train_file_counter = write_buffer_optimized(
                    train_buffer, True, train_file_counter
                )
                train_buffer = []

            if len(test_buffer) >= target_file_size:
                test_file_counter = write_buffer_optimized(
                    test_buffer, False, test_file_counter
                )
                test_buffer = []

            # Clean up memory
            del old_df, new_df

        # Force garbage collection after each batch
        gc.collect()

    # Write remaining buffers
    if train_buffer:
        train_file_counter = write_buffer_optimized(
            train_buffer, True, train_file_counter
        )
    if test_buffer:
        test_file_counter = write_buffer_optimized(
            test_buffer, False, test_file_counter
        )

    # Save sample counts
    with open(SAMPLE_COUNTS_PATH, "wb") as f:
        pickle.dump(
            {
                "train": train_count,
                "test": test_count,
            },
            f,
        )

    print(f"\nFinal sample counts: {train_count:,d} train, {test_count:,d} test")
    print(
        f"Created {train_file_counter} train files and {test_file_counter} test files"
    )


# Keep the original function but replace it with the optimized version
def prepare_data(
    input_files,
    output_dir,
    champion_encoder: LabelEncoder,
    task_means,
    task_stds,
    target_file_size: int = 50000,
    next_patch_test: bool = False,
):
    """Wrapper that calls the optimized version"""
    return prepare_data_optimized(
        input_files,
        output_dir,
        champion_encoder,
        task_means,
        task_stds,
        target_file_size,
        next_patch_test,
    )


def process_file_complete(
    args: Tuple[str, Dict[float, int], str, bool, LabelEncoder, Dict, Dict, bool],
) -> Tuple[bool, Dict[str, int], str, int, int]:
    """Process a single file completely and save directly to train or test directory"""
    (
        file_path,
        patch_mapping,
        output_dir,
        next_patch_test,
        champion_encoder,
        task_means,
        task_stds,
        is_train_file,
    ) = args

    try:
        # Read the raw file
        df = pd.read_parquet(file_path)
        original_count = len(df)

        if len(df) <= 1:
            return (
                False,
                {"total_raw": original_count, "final_count": 0},
                file_path,
                0,
                0,
            )

        # Filter out games where matchId prefix doesn't match region(to fix duplicate rows, because of an resolved issue in data collection)
        df["match_prefix"] = df["matchId"].str.split("_").str[0]
        region_match_mask = df["match_prefix"] == df["region"]
        df = df[region_match_mask]
        df.drop("match_prefix", axis=1, inplace=True)

        if len(df) == 0:
            return (
                False,
                {
                    "total_raw": original_count,
                    "region_mismatch": original_count,
                    "final_count": 0,
                },
                file_path,
                0,
                0,
            )

        # Calculate raw patch numbers and filter
        raw_patches = get_patch_from_raw_data(df)
        df["patch"] = raw_patches.map(lambda x: patch_mapping.get(x, 0))
        df = df[df["patch"] > 0]

        if len(df) == 0:
            return (
                False,
                {
                    "total_raw": original_count,
                    "patch_filtered": len(df),
                    "final_count": 0,
                },
                file_path,
                0,
                0,
            )

        # Apply filters
        df, filter_counts = filter_outliers(df)

        if len(df) <= 1:
            return (False, {**filter_counts, "final_count": len(df)}, file_path, 0, 0)

        # Apply all getters to create new columns
        for col, col_def in COLUMNS.items():
            if col_def.getter is not None:
                df[col] = col_def.getter(df)

        for task, task_def in TASKS.items():
            if task_def.getter is not None:
                df[task] = task_def.getter(df)

        # Now transform the data for ML (encoders, normalization, etc.)
        new_data = {}

        # Copy categorical columns with proper dtype
        for col in KNOWN_CATEGORICAL_COLUMNS_NAMES:
            if col in df.columns:
                new_data[col] = df[col].astype("int32", copy=False)

        # Handle patch column
        new_data["patch"] = df["patch"].astype("int32", copy=False)

        # Process champion_ids
        if "champion_ids" in df.columns:
            new_data["champion_ids"] = df["champion_ids"].apply(
                lambda x: champion_encoder.transform(x)
            )

        # Process regression tasks
        regression_tasks = [
            task
            for task, task_def in TASKS.items()
            if task_def.task_type == TaskType.REGRESSION
        ]
        for task in regression_tasks:
            if task in df.columns:
                if task_stds[task] != 0:
                    new_data[task] = (
                        (df[task] - task_means[task]) / task_stds[task]
                    ).astype("float32", copy=False)
                else:
                    new_data[task] = (df[task] - task_means[task]).astype(
                        "float32", copy=False
                    )

        # Process binary classification tasks
        binary_tasks = [
            task
            for task, task_def in TASKS.items()
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION
            and not task.startswith("win_prediction_")
        ]
        for task in binary_tasks:
            if task == "win_prediction":
                new_data[task] = df["team_100_win"].astype("float32", copy=False)
            elif task in df.columns:
                new_data[task] = df[task].astype("float32", copy=False)

        # Create final DataFrame
        final_df = pd.DataFrame(new_data)

        # Handle train/test split
        train_count = 0
        test_count = 0

        if next_patch_test:
            # Use patch-based splitting for next_patch_test
            raw_patches = get_patch_from_raw_data(df)

            # Get recent patches from the patch mapping we already have
            all_patches = sorted(patch_mapping.keys())
            recent_patches = all_patches[-2:]

            if any(patch in recent_patches for patch in raw_patches.unique()):
                is_recent_patch = raw_patches.isin(recent_patches)

                if is_recent_patch.any():
                    test_df = final_df[is_recent_patch]
                    test_output_path = os.path.join(
                        output_dir, "test", os.path.basename(file_path)
                    )
                    test_df.to_parquet(
                        test_output_path,
                        index=False,
                        compression="snappy",
                        engine="pyarrow",
                    )
                    test_count = len(test_df)

                if (~is_recent_patch).any():
                    train_df = final_df[~is_recent_patch]
                    train_output_path = os.path.join(
                        output_dir, "train", os.path.basename(file_path)
                    )
                    train_df.to_parquet(
                        train_output_path,
                        index=False,
                        compression="snappy",
                        engine="pyarrow",
                    )
                    train_count = len(train_df)
            else:
                # All data goes to train
                train_output_path = os.path.join(
                    output_dir, "train", os.path.basename(file_path)
                )
                final_df.to_parquet(
                    train_output_path,
                    index=False,
                    compression="snappy",
                    engine="pyarrow",
                )
                train_count = len(final_df)
        else:
            # File-level train/test split
            if is_train_file:
                train_output_path = os.path.join(
                    output_dir, "train", os.path.basename(file_path)
                )
                final_df.to_parquet(
                    train_output_path,
                    index=False,
                    compression="snappy",
                    engine="pyarrow",
                )
                train_count = len(final_df)
            else:
                test_output_path = os.path.join(
                    output_dir, "test", os.path.basename(file_path)
                )
                final_df.to_parquet(
                    test_output_path,
                    index=False,
                    compression="snappy",
                    engine="pyarrow",
                )
                test_count = len(final_df)

        # Clean up memory
        del df, final_df
        gc.collect()

        return (
            True,
            {**filter_counts, "final_count": train_count + test_count},
            file_path,
            train_count,
            test_count,
        )

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        gc.collect()
        return (
            False,
            {"total_raw": 0, "error": str(e), "final_count": 0},
            file_path,
            0,
            0,
        )


def prepare_data_single_pass(
    input_files,
    output_dir,
    champion_encoder: LabelEncoder,
    task_means,
    task_stds,
    next_patch_test: bool = False,
):
    """Ultra-fast single-pass data preparation with file-level train/test split"""
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Clean up existing files
    for folder in ["train", "test"]:
        for file_path in glob.glob(os.path.join(output_dir, folder, "*.parquet")):
            os.remove(file_path)

    # Load patch mapping
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]

    # Prepare arguments for parallel processing
    all_args = []

    if next_patch_test:
        # For next_patch_test, process all files (splitting happens within files by patch)
        print("Using patch-based train/test split (processing all files)")
        for file_path in input_files:
            all_args.append(
                (
                    file_path,
                    patch_mapping,
                    output_dir,
                    next_patch_test,
                    champion_encoder,
                    task_means,
                    task_stds,
                    True,  # dummy value, not used for next_patch_test
                )
            )
    else:
        # For normal mode, do file-level train/test split
        rng = np.random.default_rng(42)  # Reproducible
        shuffled_files = input_files.copy()
        rng.shuffle(shuffled_files)

        # 90/10 split at file level
        train_file_count = int(len(shuffled_files) * 0.9)
        train_files = shuffled_files[:train_file_count]
        test_files = shuffled_files[train_file_count:]

        print(
            f"File-level split: {len(train_files)} train files, {len(test_files)} test files"
        )

        # Add train files
        for file_path in train_files:
            all_args.append(
                (
                    file_path,
                    patch_mapping,
                    output_dir,
                    next_patch_test,
                    champion_encoder,
                    task_means,
                    task_stds,
                    True,  # is_train_file
                )
            )

        # Add test files
        for file_path in test_files:
            all_args.append(
                (
                    file_path,
                    patch_mapping,
                    output_dir,
                    next_patch_test,
                    champion_encoder,
                    task_means,
                    task_stds,
                    False,  # is_train_file
                )
            )

    # Process all files in parallel
    num_cores = max(1, mp.cpu_count() - 1)
    chunk_size = max(1, len(all_args) // (num_cores * 4))

    train_count = 0
    test_count = 0
    cumulative_stats = {
        "total_raw": 0,
        "region_mismatch": 0,
        "patch_filtered": 0,
        "processed_in_patch": 0,
        "filtered": 0,
        "gold_filter": 0,
        "deaths_filter": 0,
        "cs_filter_normal": 0,
        "level_filter": 0,
        "errors": 0,
    }

    with mp.Pool(num_cores) as pool:
        # Process in batches for memory management
        batch_size = chunk_size * 5
        for i in range(0, len(all_args), batch_size):
            batch = all_args[i : i + batch_size]

            results = list(
                tqdm(
                    pool.imap(process_file_complete, batch, chunksize=chunk_size),
                    total=len(batch),
                    desc=f"Processing batch {i//batch_size + 1}/{(len(all_args) + batch_size - 1)//batch_size}",
                )
            )

            # Aggregate results
            for (
                success,
                filter_counts,
                file_path,
                file_train_count,
                file_test_count,
            ) in results:
                cumulative_stats["total_raw"] += filter_counts.get("total_raw", 0)
                cumulative_stats["region_mismatch"] += filter_counts.get(
                    "region_mismatch", 0
                )

                if success:
                    train_count += file_train_count
                    test_count += file_test_count
                    cumulative_stats["processed_in_patch"] += filter_counts.get(
                        "final_count", 0
                    )

                    # Add other filter stats
                    for key in [
                        "gold_filter",
                        "deaths_filter",
                        "cs_filter_normal",
                        "level_filter",
                    ]:
                        cumulative_stats[key] += filter_counts.get(key, 0)

                if "error" in filter_counts:
                    cumulative_stats["errors"] += 1

            # Garbage collection between batches
            gc.collect()

    # Save sample counts
    with open(SAMPLE_COUNTS_PATH, "wb") as f:
        pickle.dump({"train": train_count, "test": test_count}, f)

    print(f"\nFinal sample counts: {train_count:,d} train, {test_count:,d} test")
    print(
        f"Successfully processed {len([f for f in os.listdir(os.path.join(output_dir, 'train'))])} train files"
    )
    print(
        f"Successfully processed {len([f for f in os.listdir(os.path.join(output_dir, 'test'))])} test files"
    )

    # Print filtering stats
    if cumulative_stats["total_raw"] > 0:
        final_summary = [
            f"\nFinal filtering statistics:",
            f"├─ Total raw games: {cumulative_stats['total_raw']:,d}",
            f"├─ Region mismatch filtered: {cumulative_stats['region_mismatch']:,d}",
            f"├─ Games in recent patches: {cumulative_stats['processed_in_patch']:,d}",
            f"├─ Gold filter: {cumulative_stats['gold_filter']:,d} games",
            f"├─ Deaths filter: {cumulative_stats['deaths_filter']:,d} games",
            f"├─ CS filter: {cumulative_stats['cs_filter_normal']:,d} games",
            f"├─ Level filter: {cumulative_stats['level_filter']:,d} games",
            f"├─ Errors encountered: {cumulative_stats['errors']:,d}",
            f"└─ Final games kept: {train_count + test_count:,d}",
        ]
        tqdm.write("\n".join(final_summary))


def extract_champion_ids(file_path: str) -> set:
    """Extract unique champion IDs from a single raw file"""
    try:
        # Read only the champion ID columns we need
        df = pd.read_parquet(
            file_path,
            columns=[
                f"team_{team}_{role}_championId"
                for team in [100, 200]
                for role in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
            ],
        )
        # Extract all unique champion IDs
        champion_ids = set()
        for col in df.columns:
            if col.endswith("_championId"):
                champion_ids.update(df[col].dropna().astype(int).unique())
        return champion_ids
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()


def process_file_for_stats(
    file_path: str,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Process a single file to compute task statistics"""
    try:
        # Read the raw file
        df = pd.read_parquet(file_path)

        # Apply basic filtering (same as in main processing)
        df["match_prefix"] = df["matchId"].str.split("_").str[0]
        region_match_mask = df["match_prefix"] == df["region"]
        df = df[region_match_mask]
        df.drop("match_prefix", axis=1, inplace=True)

        if len(df) == 0:
            return {}, {}, 0

        # Apply filters
        df, _ = filter_outliers(df)

        if len(df) <= 1:
            return {}, {}, 0

        # Apply all getters to create computed columns (needed for task stats)
        for col, col_def in COLUMNS.items():
            if col_def.getter is not None:
                df[col] = col_def.getter(df)

        for task, task_def in TASKS.items():
            if task_def.getter is not None:
                df[task] = task_def.getter(df)

        # Compute statistics for regression tasks
        task_sums = {}
        task_sumsq = {}

        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION and task in df.columns:
                task_sums[task] = df[task].sum()
                task_sumsq[task] = (df[task] ** 2).sum()

        return task_sums, task_sumsq, len(df)

    except Exception as e:
        print(f"Error processing {file_path} for stats: {e}")
        return {}, {}, 0


def create_champion_id_encoder_from_raw(data_files: List[str]) -> LabelEncoder:
    """Create encoder for champion_ids from raw files using parallel processing"""
    print(f"Creating encoder for champion_ids from raw files")

    # Use multiprocessing to process files in parallel
    num_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(extract_champion_ids, data_files),
                total=len(data_files),
                desc="Extracting champion_ids",
            )
        )

    # Combine all unique values
    unique_values = set()
    for result in results:
        unique_values.update(result)

    unique_values = sorted(unique_values)
    unique_values.append("UNKNOWN")

    encoder = LabelEncoder()
    encoder.fit(unique_values)
    return encoder


def compute_task_stats_from_raw(
    data_files: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute task statistics from raw files using parallel processing"""
    print("Computing task statistics from raw files")

    # Use multiprocessing with a subset of files for efficiency
    sample_size = min(len(data_files), 1000)  # Use up to 1000 files for stats
    sample_files = data_files[:sample_size]

    num_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(process_file_for_stats, sample_files),
                total=len(sample_files),
                desc="Computing task statistics",
            )
        )

    # Combine results
    total_task_sums = {}
    total_task_sumsq = {}
    total_count = 0

    for task_sums, task_sumsq, count in results:
        for task in task_sums:
            if task not in total_task_sums:
                total_task_sums[task] = 0
                total_task_sumsq[task] = 0
            total_task_sums[task] += task_sums[task]
            total_task_sumsq[task] += task_sumsq[task]
        total_count += count

    if total_count == 0:
        raise ValueError("No valid data found for computing task statistics")

    # Compute final statistics
    task_means = {task: total_task_sums[task] / total_count for task in total_task_sums}
    task_stds = {
        task: np.sqrt((total_task_sumsq[task] / total_count) - (task_means[task] ** 2))
        for task in total_task_sums
    }

    return task_means, task_stds


def main():
    parser = argparse.ArgumentParser(description="Prepare data for machine learning")
    parser.add_argument(
        "--input-dir", default=RAW_AZURE_DIR, help="Input directory containing raw data"
    )
    parser.add_argument(
        "--output-dir",
        default=PREPARED_DATA_DIR,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help="Skip filtering of outliers/griefers",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip task statistics calculation and use existing files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limits to 100 files and enables verbose output)",
    )
    parser.add_argument(
        "--next-patch-test",
        action="store_true",
        help="Create test set from 2 newest patches, mapping them to 3rd newest patch",
    )
    args = parser.parse_args()

    # Set global DEBUG flag based on args
    global DEBUG
    DEBUG = args.debug

    input_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))

    if DEBUG:
        print(f"Debug mode enabled - limiting to {DEBUG_FILE_LIMIT} files")
        input_files = input_files[:DEBUG_FILE_LIMIT]
        print(f"Processing {len(input_files)} files")

    # Step 1: Compute patch mapping (still needed for filtering)
    print("Computing patch mapping...")
    patch_mapping, patch_counts = compute_patch_mapping(
        input_files, args.next_patch_test
    )

    # Save the patch mapping and counts for future reference
    with open(PATCH_MAPPING_PATH, "wb") as f:
        pickle.dump(
            {
                "mapping": patch_mapping,
                "counts": patch_counts,
                "next_patch_test": args.next_patch_test,
            },
            f,
        )

    # Step 2: Create encoders from raw files
    print("Creating encoders from raw files...")
    champion_encoder = create_champion_id_encoder_from_raw(input_files)
    print("Saving encoders...")
    with open(CHAMPION_ID_ENCODER_PATH, "wb") as f:
        pickle.dump({"mapping": champion_encoder}, f)

    # Step 3: Compute task statistics from raw files (if not skipping)
    if args.skip_stats:
        print("Loading existing task statistics...")
        try:
            with open(TASK_STATS_PATH, "rb") as f:
                stats_data = pickle.load(f)
                task_means = stats_data["means"]
                task_stds = stats_data["stds"]
        except FileNotFoundError:
            raise ValueError(
                f"Could not find existing task statistics at {TASK_STATS_PATH}. "
                "Please run without --skip-stats first to generate the statistics."
            )
    else:
        print("Computing statistics from raw files...")
        task_means, task_stds = compute_task_stats_from_raw(input_files)
        print("Saving task statistics...")
        with open(TASK_STATS_PATH, "wb") as f:
            pickle.dump({"means": task_means, "stds": task_stds}, f)

    # Step 4: Single-pass processing (no temporary directory!)
    print("Processing files in single pass...")
    prepare_data_single_pass(
        input_files,
        args.output_dir,
        champion_encoder,
        task_means,
        task_stds,
        next_patch_test=args.next_patch_test,
    )

    print("Data preparation completed!")


if __name__ == "__main__":
    main()
