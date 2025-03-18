# scripts/match-prediction/prepare_data.py
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
    CHAMPION_ID_MAPPING_PATH,
    TASK_STATS_PATH,
    PATCH_MAPPING_PATH,
    SAMPLE_COUNTS_PATH,
)
from utils.match_prediction.column_definitions import (
    COLUMNS,
    KNOWN_CATEGORICAL_COLUMNS,
    get_patch_from_raw_data,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from typing import List, Dict, Tuple
from collections import defaultdict

# Constants
NUM_RECENT_PATCHES = 23  # Number of most recent patches to use for training
PATCH_MIN_GAMES = 200_000  # Minimum number of games in a patch to be considered standalone(otherwise merged with previous patch)
DEBUG = False
DEBUG_FILE_LIMIT = 100  # Maximum number of files to process in debug mode

# Filter constants
MIN_GOLD_15MIN = 2400  # Minimum gold at 15 minutes
MAX_DEATHS_15MIN = 10  # Maximum deaths at 15 minutes
MIN_CS_15MIN_NORMAL = 25  # Minimum CS at 15 minutes for non-support roles
MIN_LEVEL_15MIN = 4  # Minimum level at 15 minutes


def create_champion_id_mapping(data_files):
    print(f"Creating encoder for champion_ids")
    unique_values = set()
    for file_path in tqdm(data_files, desc="Processing champion_ids"):
        df = pd.read_parquet(file_path)
        unique_values.update(df["champion_ids"].explode().unique())

    unique_values = sorted(unique_values)
    unique_values.append("UNKNOWN")

    encoder = LabelEncoder()
    encoder.fit(unique_values)

    return encoder


def compute_task_stats(data_files):
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
    total_count = 0

    for file_path in tqdm(data_files, desc="Computing stats"):
        df = pd.read_parquet(file_path)

        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION:
                task_sums[task] += df[task].sum()
                task_sumsq[task] += (df[task] ** 2).sum()
        total_count += len(df)

    task_means = {task: task_sums[task] / total_count for task in task_sums}
    task_stds = {
        task: np.sqrt((task_sumsq[task] / total_count) - (task_means[task] ** 2))
        for task in task_sums
    }

    return task_means, task_stds


def compute_patch_mapping(input_files: List[str]) -> Dict[float, int]:
    """
    Analyze all files to create a mapping for patch numbers.
    Takes the NUM_RECENT_PATCHES most recent patches, mapping any patch with <200k games to the previous patch.
    Returns a dictionary mapping original patch numbers to normalized ones (1-NUM_RECENT_PATCHES).
    """
    patch_counts = defaultdict(int)

    print("Computing patch distribution...")
    for file_path in tqdm(input_files):
        df = pd.read_parquet(file_path)
        raw_patches = get_patch_from_raw_data(df)
        for patch in raw_patches:
            patch_counts[patch] += 1
    # All patches in chronological order, works because patches are 0 padded
    # "14.01" < "14.02" < "14.10" in lexicographical order.
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
    normalized_value = 1
    previous_significant_patch = None

    # Process patches from oldest to newest of the recent patches
    for patch in recent_patches:
        if (
            patch_counts[patch] > PATCH_MIN_GAMES or DEBUG
        ):  # In debug mode, accept all patches
            # This is a significant patch
            patch_mapping[patch] = normalized_value
            previous_significant_patch = patch
            normalized_value += 1
        elif previous_significant_patch is not None:
            # Map to previous significant patch
            patch_mapping[patch] = patch_mapping[previous_significant_patch]
        else:
            # First patch has low data
            if DEBUG:
                # In debug mode, accept it anyway
                patch_mapping[patch] = normalized_value
                previous_significant_patch = patch
                normalized_value += 1
            else:
                raise ValueError(f"First patch {patch} has insufficient data!")

    # Log the mapping for transparency
    print("\nPatch mapping (last NUM_RECENT_PATCHES patches):")
    for patch in sorted(patch_mapping.keys()):
        mapped_value = patch_mapping[patch]
        print(f"Patch {patch} -> {mapped_value} ({patch_counts[patch]} games)")

    return patch_mapping, patch_counts


def filter_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter out games with potential griefers/AFKs based on specified rules.
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

    # 1. Gold filter
    gold_filter_conditions = []
    for role in roles:
        for team in teams:
            col = f"team_{team}_{role}_totalGold_at_900000"
            if col in df.columns:
                condition = df[col] < MIN_GOLD_15MIN
                gold_filter_conditions.append(condition)

    if gold_filter_conditions:
        gold_filter = pd.concat(gold_filter_conditions, axis=1).any(axis=1)
        filter_counts["gold_filter"] = gold_filter.sum()
        df = df[~gold_filter]

    # 2. Deaths filter
    deaths_filter_conditions = []
    for role in roles:
        for team in teams:
            col = f"team_{team}_{role}_deaths_at_900000"
            if col in df.columns:
                condition = df[col] > MAX_DEATHS_15MIN
                deaths_filter_conditions.append(condition)

    if deaths_filter_conditions:
        deaths_filter = pd.concat(deaths_filter_conditions, axis=1).any(axis=1)
        filter_counts["deaths_filter"] = deaths_filter.sum()
        df = df[~deaths_filter]

    # 3. CS filter
    cs_filter_normal_conditions = []
    for role in roles:
        if role != "UTILITY":
            for team in teams:
                col = f"team_{team}_{role}_creepScore_at_900000"
                if col in df.columns:
                    condition = df[col] < MIN_CS_15MIN_NORMAL
                    cs_filter_normal_conditions.append(condition)

    if cs_filter_normal_conditions:
        cs_filter_normal = pd.concat(cs_filter_normal_conditions, axis=1).any(axis=1)
        filter_counts["cs_filter_normal"] = cs_filter_normal.sum()
        df = df[~cs_filter_normal]

    # 4. Level filter
    level_filter_conditions = []
    for role in roles:
        for team in teams:
            col = f"team_{team}_{role}_level_at_900000"
            if col in df.columns:
                condition = df[col] < MIN_LEVEL_15MIN
                level_filter_conditions.append(condition)

    if level_filter_conditions:
        level_filter = pd.concat(level_filter_conditions, axis=1).any(axis=1)
        filter_counts["level_filter"] = level_filter.sum()
        df = df[~level_filter]

    # Print summary using tqdm.write
    total_filtered = original_count - len(df)
    summary = [
        f"Filtering summary:",
        f"├─ Gold filter: {filter_counts['gold_filter']:,d} rows",
        f"├─ Deaths filter: {filter_counts['deaths_filter']:,d} rows",
        f"├─ CS filter: {filter_counts['cs_filter_normal']:,d} rows",
        f"├─ Level filter: {filter_counts['level_filter']:,d} rows",
        f"└─ Total: {total_filtered:,d} rows ({total_filtered/original_count*100:.1f}%)",
    ]
    if DEBUG:
        tqdm.write("\n".join(summary))

    return df, filter_counts


def prepare_data(
    input_files,
    output_dir,
    champion_mapping,
    task_means,
    task_stds,
    target_file_size: int = 50000,  # Target number of rows per output file
):
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Clean up existing files
    for folder in ["train", "test"]:
        for file_path in glob.glob(os.path.join(output_dir, folder, "*.parquet")):
            os.remove(file_path)

    train_buffer = []
    test_buffer = []
    train_file_counter = 0
    test_file_counter = 0
    train_count = 0
    test_count = 0

    def write_buffer(buffer, is_train: bool, file_counter: int):
        if not buffer:
            return file_counter

        file_prefix = "train" if is_train else "test"
        output_path = os.path.join(
            output_dir, file_prefix, f"{file_prefix}_{file_counter:04d}.parquet"
        )

        df = pd.DataFrame(buffer)
        df.to_parquet(output_path, index=False)
        return file_counter + 1

    # Shuffle input files for better data distribution
    rng = np.random.default_rng(42)  # Use seeded RNG for reproducibility
    rng.shuffle(input_files)

    for file_index, file_path in enumerate(tqdm(input_files, desc="Preparing data")):
        old_df = pd.read_parquet(file_path)

        # Create empty DataFrame with all columns pre-allocated with correct types
        column_dtypes = {
            # Categorical columns are integers
            **{col: "int32" for col in KNOWN_CATEGORICAL_COLUMNS},
            # Special columns
            "patch": "int32",
            "champion_ids": "object",  # array of ints needs object dtype
            # All tasks are floats
            **{task: "float32" for task in TASKS.keys()},
        }

        new_df = pd.DataFrame(
            # Initialize with NaN for float columns, 0 for int columns
            {
                col: np.zeros(len(old_df), dtype=dtype)
                for col, dtype in column_dtypes.items()
            }
        )

        if len(old_df) <= 1:
            print(
                f"Skipping file {file_path} - insufficient samples ({len(old_df)} rows)"
            )
            continue

        # Can just copy, it was already processed(because categories are known)
        new_df[KNOWN_CATEGORICAL_COLUMNS] = old_df[KNOWN_CATEGORICAL_COLUMNS].astype(
            "int32"
        )
        # Special handling columns(patch and champion ids)
        # Patch was already mapped, just copy
        new_df["patch"] = old_df["patch"].astype("int32")
        # Champion mapping was just created, need to apply it
        new_df["champion_ids"] = old_df["champion_ids"].apply(
            lambda x: champion_mapping.transform(x)
        )
        # Process all numerical tasks at once
        regression_tasks = [
            task
            for task, task_def in TASKS.items()
            if task_def.task_type == TaskType.REGRESSION
        ]

        # Vectorized operation for all regression tasks
        for task in regression_tasks:
            if task_stds[task] != 0:
                new_df[task] = (
                    (old_df[task] - task_means[task]) / task_stds[task]
                ).astype("float32")
            else:
                new_df[task] = (old_df[task] - task_means[task]).astype("float32")

        # Split into train/test
        if len(new_df) < 10:
            df_train = new_df.iloc[:-1]
            df_test = new_df.iloc[-1:]
        else:
            df_train, df_test = train_test_split(new_df, test_size=0.1, random_state=42)

        # Add to buffers
        train_buffer.extend(df_train.to_dict("records"))
        test_buffer.extend(df_test.to_dict("records"))

        train_count += len(df_train)
        test_count += len(df_test)

        # Write buffers if they exceed target size
        if len(train_buffer) >= target_file_size:
            train_file_counter = write_buffer(train_buffer, True, train_file_counter)
            train_buffer = []

        if len(test_buffer) >= target_file_size:
            test_file_counter = write_buffer(test_buffer, False, test_file_counter)
            test_buffer = []

        # Clear memory
        del old_df, new_df, df_train, df_test

    # Write remaining buffers
    if train_buffer:
        train_file_counter = write_buffer(train_buffer, True, train_file_counter)
    if test_buffer:
        test_file_counter = write_buffer(test_buffer, False, test_file_counter)

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


def add_computed_columns(input_files: List[str], output_dir: str) -> List[str]:
    """
    Add computed columns to parquet files and save them to a new directory.
    Only includes games from the last NUM_RECENT_PATCHES patches.
    Returns the list of new file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    new_files = []

    # First, compute the patch mapping from the entire dataset
    patch_mapping, patch_counts = compute_patch_mapping(input_files)

    # Save the patch mapping and counts for future reference
    with open(PATCH_MAPPING_PATH, "wb") as f:
        pickle.dump({"mapping": patch_mapping, "counts": patch_counts}, f)

    # Track cumulative stats
    cumulative_stats = {
        "total_raw": 0,  # Total games before patch filtering
        "patch_filtered": 0,  # Games filtered due to old patches
        "processed_in_patch": 0,  # Games processed after patch filtering
        "filtered": 0,
        "gold_filter": 0,
        "deaths_filter": 0,
        "cs_filter_normal": 0,
        "level_filter": 0,
    }

    pbar = tqdm(input_files, desc="Adding computed columns")
    for file_path in pbar:
        df = pd.read_parquet(file_path)
        original_count = len(df)
        cumulative_stats["total_raw"] += original_count

        # Calculate raw patch numbers
        raw_patches = get_patch_from_raw_data(df)

        # Filter for games only in the patch mapping (last NUM_RECENT_PATCHES patches)
        df["patch"] = raw_patches.map(lambda x: patch_mapping.get(x, 0))
        df = df[df["patch"] > 0]  # Remove games from old patches

        # Track how many games were filtered due to patches
        cumulative_stats["patch_filtered"] += original_count - len(df)

        # Skip empty dataframes
        if len(df) == 0:
            continue

        # Track games that are in the valid patch range
        games_in_patch = len(df)
        cumulative_stats["processed_in_patch"] += games_in_patch

        df, filter_counts = filter_outliers(df)

        # Update cumulative stats (only for games within patch range)
        cumulative_stats["filtered"] += games_in_patch - len(df)
        for key in ["gold_filter", "deaths_filter", "cs_filter_normal", "level_filter"]:
            cumulative_stats[key] += filter_counts[key]

        # Update progress bar description with cumulative stats
        filtered_pct = (
            (
                cumulative_stats["filtered"]
                / cumulative_stats["processed_in_patch"]
                * 100
            )
            if cumulative_stats["processed_in_patch"] > 0
            else 0
        )
        pbar.set_description(
            f"Processed (in patch): {cumulative_stats['processed_in_patch']:,d} | Filtered: {filtered_pct:.1f}%"
        )

        # Skip if no data left after filtering
        if len(df) <= 1:
            print(
                f"Skipping file {file_path} after filtering - insufficient samples ({len(df)} rows)"
            )
            continue

        # Apply all getters to create new columns
        for col, col_def in COLUMNS.items():
            if col_def.getter is not None:
                df[col] = col_def.getter(df)
        for task, task_def in TASKS.items():
            if task_def.getter is not None:
                df[task] = task_def.getter(df)

        # Save the enhanced dataframe
        new_file_path = os.path.join(output_dir, os.path.basename(file_path))
        df.to_parquet(new_file_path, index=False)
        new_files.append(new_file_path)

        # Clear memory
        del df

    # Print final cumulative stats using tqdm.write
    if cumulative_stats["total_raw"] > 0:
        final_summary = [
            f"\nFinal filtering statistics:",
            f"├─ Total raw games: {cumulative_stats['total_raw']:,d}",
            f"├─ Games from old patches: {cumulative_stats['patch_filtered']:,d} ({cumulative_stats['patch_filtered']/cumulative_stats['total_raw']*100:.1f}%)",
            f"├─ Games in recent patches: {cumulative_stats['processed_in_patch']:,d}",
            f"├─ Filtering results (for games in recent patches only):",
            f"│  ├─ Gold filter: {cumulative_stats['gold_filter']:,d} games ({cumulative_stats['gold_filter']/cumulative_stats['processed_in_patch']*100:.1f}%)",
            f"│  ├─ Deaths filter: {cumulative_stats['deaths_filter']:,d} games ({cumulative_stats['deaths_filter']/cumulative_stats['processed_in_patch']*100:.1f}%)",
            f"│  ├─ CS filter: {cumulative_stats['cs_filter_normal']:,d} games ({cumulative_stats['cs_filter_normal']/cumulative_stats['processed_in_patch']*100:.1f}%)",
            f"│  ├─ Level filter: {cumulative_stats['level_filter']:,d} games ({cumulative_stats['level_filter']/cumulative_stats['processed_in_patch']*100:.1f}%)",
            f"│  └─ Total filtered: {cumulative_stats['filtered']:,d} games ({cumulative_stats['filtered']/cumulative_stats['processed_in_patch']*100:.1f}%)",
            f"└─ Final games kept: {cumulative_stats['processed_in_patch'] - cumulative_stats['filtered']:,d}",
        ]
        tqdm.write("\n".join(final_summary))

    return new_files


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
        "--debug",
        action="store_true",
        help="Enable debug mode (limits to 100 files and enables verbose output)",
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

    # Create temporary directory for intermediate files
    temp_dir = os.path.join(args.output_dir, "temp")

    print("Adding computed columns...")
    enhanced_files = add_computed_columns(input_files, temp_dir)

    print("Creating encoders...")
    champion_mapping = create_champion_id_mapping(enhanced_files)

    print("Computing statistics...")
    task_means, task_stds = compute_task_stats(enhanced_files)

    print("Saving encoders and statistics...")
    with open(CHAMPION_ID_MAPPING_PATH, "wb") as f:
        pickle.dump(champion_mapping, f)
    with open(TASK_STATS_PATH, "wb") as f:
        pickle.dump({"means": task_means, "stds": task_stds}, f)

    print("Preparing final data...")
    prepare_data(
        enhanced_files,
        args.output_dir,
        champion_mapping,
        task_means,
        task_stds,
    )

    # Clean up temporary files
    import shutil

    shutil.rmtree(temp_dir)

    print("Data preparation completed.")


if __name__ == "__main__":
    main()
