# scripts/match-prediction/prepare-data.py
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
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    TASK_STATS_PATH,
)
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    COLUMNS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from typing import List, Dict
from collections import defaultdict


def load_data(file_path):
    return pd.read_parquet(file_path)


def save_data(df, file_path):
    df.to_parquet(file_path, index=False)


def create_encoders(data_files):
    encoders = {}
    for col in CATEGORICAL_COLUMNS + ["champion_ids"]:
        print(f"Creating encoder for {col}")
        unique_values = set()
        for file_path in tqdm(data_files, desc=f"Processing {col}"):
            df = load_data(file_path)
            if col == "champion_ids":
                unique_values.update(df[col].explode().unique())
            else:
                unique_values.update(df[col].unique())

        unique_values = sorted(unique_values)
        unique_values.append("UNKNOWN")  # Add UNKNOWN for unseen categories

        encoder = LabelEncoder()
        encoder.fit(unique_values)
        encoders[col] = encoder

    return encoders


def compute_stats(data_files):
    numerical_sums = {col: 0.0 for col in NUMERICAL_COLUMNS}
    numerical_sumsq = {col: 0.0 for col in NUMERICAL_COLUMNS}
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
        df = load_data(file_path)
        for col in NUMERICAL_COLUMNS:
            numerical_sums[col] += df[col].sum()
            numerical_sumsq[col] += (df[col] ** 2).sum()
        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION:
                task_sums[task] += df[task].sum()
                task_sumsq[task] += (df[task] ** 2).sum()
        total_count += len(df)

    numerical_means = {
        col: numerical_sums[col] / total_count for col in NUMERICAL_COLUMNS
    }
    numerical_stds = {
        col: np.sqrt((numerical_sumsq[col] / total_count) - (numerical_means[col] ** 2))
        for col in NUMERICAL_COLUMNS
    }
    task_means = {task: task_sums[task] / total_count for task in task_sums}
    task_stds = {
        task: np.sqrt((task_sumsq[task] / total_count) - (task_means[task] ** 2))
        for task in task_sums
    }

    return numerical_means, numerical_stds, task_means, task_stds


def compute_patch_mapping(input_files: List[str]) -> Dict[float, int]:
    """
    Analyze all files to create a mapping for patch numbers.
    Takes the 5 most recent patches, mapping any patch with <200k games to the previous patch.
    Returns a dictionary mapping original patch numbers to normalized ones (1-5).
    """
    patch_counts = defaultdict(int)

    print("Computing patch distribution...")
    for file_path in tqdm(input_files):
        df = load_data(file_path)
        raw_patches = df["gameVersionMajorPatch"] * 50 + df["gameVersionMinorPatch"]
        for patch in raw_patches:
            patch_counts[patch] += 1

    all_patches = sorted(patch_counts.keys())  # All patches in chronological order

    # Get the last 5 patches
    recent_patches = all_patches[-5:]
    if len(recent_patches) < 5:
        raise ValueError("Not enough patches in the dataset!")

    # Create mapping, handling low-data patches
    patch_mapping = {}
    normalized_value = 1
    previous_significant_patch = None

    # Process patches from oldest to newest of the last 5
    for patch in recent_patches:
        if patch_counts[patch] > 200000:
            # This is a significant patch
            patch_mapping[patch] = normalized_value
            previous_significant_patch = patch
            normalized_value += 1
        elif previous_significant_patch is not None:
            # Map to previous significant patch
            patch_mapping[patch] = patch_mapping[previous_significant_patch]
        else:
            # First patch has low data, shouldn't happen with 5 patches
            raise ValueError(f"First patch {patch} has insufficient data!")

    # Log the mapping for transparency
    print("\nPatch mapping (last 5 patches):")
    for patch in sorted(patch_mapping.keys()):
        mapped_value = patch_mapping[patch]
        print(f"Patch {patch:.2f} -> {mapped_value} ({patch_counts[patch]} games)")

    return patch_mapping, patch_counts


def prepare_data(
    input_files,
    output_dir,
    encoders,
    numerical_means,
    numerical_stds,
    task_means,
    task_stds,
):
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    # clean up the folders
    for folder in ["train", "test"]:
        for file_path in glob.glob(os.path.join(output_dir, folder, "*.parquet")):
            os.remove(file_path)

    train_count = 0
    test_count = 0

    for file_index, file_path in enumerate(tqdm(input_files, desc="Preparing data")):
        df = load_data(file_path)

        if len(df) <= 1:
            print(f"Skipping file {file_path} - insufficient samples ({len(df)} rows)")
            continue

        # Encode categorical columns
        for col in CATEGORICAL_COLUMNS:
            df[col] = encoders[col].transform(df[col])

        # Encode champion_ids
        df["champion_ids"] = df["champion_ids"].apply(
            lambda x: encoders["champion_ids"].transform(x)
        )

        # Normalize numerical columns
        for col in NUMERICAL_COLUMNS:
            if numerical_stds[col] != 0:
                df[col] = (df[col] - numerical_means[col]) / numerical_stds[col]
            else:
                df[col] = (
                    df[col] - numerical_means[col]
                )  # Center the data if std is zero

        # Normalize regression tasks
        for task, task_def in TASKS.items():
            if task_def.task_type == TaskType.REGRESSION:
                if task_stds[task] != 0:
                    df[task] = (df[task] - task_means[task]) / task_stds[task]
                else:
                    df[task] = (
                        df[task] - task_means[task]
                    )  # Center the data if std is zero

        # Modified split with minimum size check
        if len(df) < 10:  # Arbitrary minimum size, adjust as needed
            # For very small datasets, use a fixed split
            df_train = df.iloc[:-1]  # All but last row
            df_test = df.iloc[-1:]   # Last row
        else:
            df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

        train_count += len(df_train)
        test_count += len(df_test)

        # Save prepared data
        save_data(
            df_train, os.path.join(output_dir, "train", f"train_{file_index}.parquet")
        )
        save_data(
            df_test, os.path.join(output_dir, "test", f"test_{file_index}.parquet")
        )

        # Clear memory
        del df, df_train, df_test

    # Save sample counts
    with open(os.path.join(output_dir, "sample_counts.pkl"), "wb") as f:
        pickle.dump({"train": train_count, "test": test_count}, f)


def add_computed_columns(input_files: List[str], output_dir: str) -> List[str]:
    """
    Add computed columns to parquet files and save them to a new directory.
    Only includes games from the last 5 patches.
    Returns the list of new file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    new_files = []

    # First, compute the patch mapping from the entire dataset
    patch_mapping, patch_counts = compute_patch_mapping(input_files)

    # Save the patch mapping and counts for future reference
    with open(os.path.join(PREPARED_DATA_DIR, "patch_mapping.pkl"), "wb") as f:
        pickle.dump({"mapping": patch_mapping, "counts": patch_counts}, f)

    for file_path in tqdm(input_files, desc="Adding computed columns"):
        df = load_data(file_path)

        # Calculate raw patch numbers
        raw_patches = df["gameVersionMajorPatch"] * 50 + df["gameVersionMinorPatch"]

        # Filter for games only in the patch mapping (last 5 patches)
        df["numerical_patch"] = raw_patches.map(lambda x: patch_mapping.get(x, 0))
        df = df[df["numerical_patch"] > 0]  # Remove games from old patches

        # Skip empty dataframes
        if len(df) == 0:
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
        save_data(df, new_file_path)
        new_files.append(new_file_path)

        # Clear memory
        del df

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
    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))

    # Create temporary directory for intermediate files
    temp_dir = os.path.join(args.output_dir, "temp")

    print("Adding computed columns...")
    enhanced_files = add_computed_columns(input_files, temp_dir)

    print("Creating encoders...")
    encoders = create_encoders(enhanced_files)

    print("Computing statistics...")
    numerical_means, numerical_stds, task_means, task_stds = compute_stats(
        enhanced_files
    )

    print("Saving encoders and statistics...")
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    with open(NUMERICAL_STATS_PATH, "wb") as f:
        pickle.dump({"means": numerical_means, "stds": numerical_stds}, f)
    with open(TASK_STATS_PATH, "wb") as f:
        pickle.dump({"means": task_means, "stds": task_stds}, f)

    print("Preparing final data...")
    prepare_data(
        enhanced_files,
        args.output_dir,
        encoders,
        numerical_means,
        numerical_stds,
        task_means,
        task_stds,
    )

    # Clean up temporary files
    import shutil

    shutil.rmtree(temp_dir)

    print("Data preparation completed.")


if __name__ == "__main__":
    main()
