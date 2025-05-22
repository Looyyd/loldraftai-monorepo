# scripts/match-prediction/train_pro.py
# Finetunes the model on professional data

import os
import argparse
import torch
import wandb
import pickle
import pandas as pd
import numpy as np
import time
import copy
from typing import Optional, Dict, Any, List, Tuple, Callable, Set
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import random

from utils.match_prediction import (
    get_best_device,
    load_model_state_dict,
    RAW_PRO_GAMES_FILE,
    PATCH_MAPPING_PATH,
    CHAMPION_ID_ENCODER_PATH,
    TASK_STATS_PATH,
    POSITIONS,
    TEAMS,
)

from utils.match_prediction.train_utils import get_num_champions

from utils.match_prediction.match_dataset import MatchDataset, dataloader_config
from utils.match_prediction.model import Model
from utils.match_prediction.train_utils import (
    set_random_seeds,
    collate_fn,
    get_optimizer_grouped_parameters,
)
from utils.match_prediction.config import (
    TrainingConfig,
)
from utils.match_prediction.column_definitions import (
    RANKED_QUEUE_INDEX,
    PRO_QUEUE_INDEX,
)
from utils.match_prediction.task_definitions import (
    TaskDefinition,
    TaskType,
)
from utils.match_prediction.masking_strategies import MASKING_STRATEGIES

# Attempting to finetune on multiple tasks, could perhaps make the model "understand" the domain adaptations better
# Futhermore our ui integrates these tasks, so should probably train the model on them.
# (for example, pro games have less drastic gold leads and less kills)
FINE_TUNE_TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.995,
    ),
    # The masked win_prediction tasks tend to overfit, so we give them very very small weights.
    # They already adapt well, when win_prediction is finetuned, these aux tasks automatically perform well.
    "win_prediction_0_25": TaskDefinition(
        name="win_prediction_0_25",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.005,
    ),
    "win_prediction_25_30": TaskDefinition(
        name="win_prediction_25_30",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.005,
    ),
    "win_prediction_30_35": TaskDefinition(
        name="win_prediction_30_35",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.005,
    ),
    "win_prediction_35_inf": TaskDefinition(
        name="win_prediction_35_inf",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.005,
    ),
}

# Add gold tasks for all positions and teams
for position in POSITIONS:
    for team_id in TEAMS:
        task_name = f"team_{team_id}_{position}_totalGold_at_900000"
        FINE_TUNE_TASKS[task_name] = TaskDefinition(
            name=task_name,
            task_type=TaskType.REGRESSION,
            weight=0.01
            / (len(POSITIONS) * len(TEAMS)),  # Same weight as in final_tasks
        )


# in this file we get from a row, so it's different from column_definitions.py
def get_patch_from_raw_data(row: pd.Series) -> str:
    return (
        str(row["gameVersionMajorPatch"])
        + "."
        + str(row["gameVersionMinorPatch"]).zfill(2)
    )


class FineTuningConfig:
    """Configuration class for fine-tuning"""

    def __init__(self):
        self.num_epochs = 1000
        self.learning_rate = 8e-6  # Lower learning rate for fine-tuning
        self.weight_decay = 0.05
        self.dropout = 0.5
        self.batch_size = 1024 * 2

        # not including original data actually works well, there is no catastrphic forgetting
        # TODO; maybe delte logic to include original data?
        self.original_batch_size = 1024 * 7 * 2

        self.val_split = 0.2
        self.max_grad_norm = 1.0
        self.log_wandb = True
        self.save_checkpoints = False
        self.debug = False
        self.validation_interval = 20

        # Add new unfreezing parameters
        self.initial_frozen_layers = 0
        self.epoch_to_unfreeze = []  # Unfreeze one layer group at these epochs

        # Label smoothing options
        self.use_label_smoothing = False  # Enable label smoothing
        self.smooth_low = 0  # Value to smooth 0 labels to
        self.smooth_high = 1  # Value to smooth 1 labels to

        # Loss balancing parameters
        self.pro_loss_weight = 1.0  # Weight multiplier for pro data losses
        self.original_loss_weight = 1.0  # Weight multiplier for original data losses

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def to_dict(self):
        return {key: value for key, value in vars(self).items()}


def split_data_randomly(
    df: pd.DataFrame, val_split: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split the data into train and validation sets.

    Args:
        df: DataFrame containing the pro games data
        val_split: Desired fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    split_idx = int(len(indices) * (1 - val_split))

    train_df = df.iloc[indices[:split_idx]].reset_index(drop=True)
    val_df = df.iloc[indices[split_idx:]].reset_index(drop=True)

    print(f"Random split:")
    print(f"Training data: {len(train_df)} games")
    print(f"Validation data: {len(val_df)} games")

    return train_df, val_df


def split_data_by_patches(
    df: pd.DataFrame, val_split: float
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split the data by putting the most recent patches in the validation set until reaching desired split ratio.

    Args:
        df: DataFrame containing the pro games data
        val_split: Desired fraction of data for validation

    Returns:
        train_df: DataFrame containing training data (earlier patches)
        val_df: DataFrame containing validation data (recent patches)
        val_patches: List of patch versions in validation set
    """
    # Get patch for each game and add as column for easier manipulation
    df = df.copy()
    df["patch"] = df.apply(get_patch_from_raw_data, axis=1)

    # Get unique patches and sort them (newest first)
    patches = sorted(df["patch"].unique(), reverse=True)

    val_patches = []
    val_indices = set()

    # Keep adding patches to validation set until we reach desired split
    for patch in patches:
        # Find all games from this patch
        patch_matches = df.index[df["patch"] == patch].tolist()
        val_indices.update(patch_matches)
        val_patches.append(patch)

        # Check if we have enough validation data
        if len(val_indices) >= len(df) * val_split:
            break

    # Create train/val masks
    val_mask = df.index.isin(val_indices)
    train_mask = ~val_mask

    # Split the data
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    print(f"Patch-based split:")
    print(f"Validation patches: {', '.join(sorted(val_patches))}")
    print(f"Training patches: {', '.join(sorted(set(patches) - set(val_patches)))}")
    print(f"Training data: {len(train_df)} games")
    print(f"Validation data: {len(val_df)} games")

    return train_df, val_df, val_patches


class ProMatchDataset(Dataset):
    """Dataset for professional matches fine-tuning"""

    def __init__(
        self,
        pro_games_df: pd.DataFrame,
        patch_mapping: Dict[str, int],
        use_label_smoothing: bool = True,
        smooth_low: float = 0.1,
        smooth_high: float = 0.9,
        masking_function: Optional[Callable[[], int]] = None,
        unknown_champion_id: Optional[int] = None,
    ):
        with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
            self.champion_id_encoder = pickle.load(f)["mapping"]

        # Load task statistics for normalization
        with open(TASK_STATS_PATH, "rb") as f:
            task_stats = pickle.load(f)
            self.task_means = task_stats["means"]
            self.task_stds = task_stats["stds"]

        self.patch_mapping = patch_mapping
        self.df = pro_games_df
        self.use_label_smoothing = use_label_smoothing
        self.smooth_low = smooth_low
        self.smooth_high = smooth_high
        self.masking_function = masking_function
        self.unknown_champion_id = unknown_champion_id

    def __len__(self):
        return len(self.df)

    def _mask_champions(self, champion_list: List[int], num_to_mask: int) -> List[int]:
        """Masks a specific number of champions in the list with unknown_champion_id

        Args:
            champion_list: List of already encoded champion IDs
            num_to_mask: Number of champions to mask
        """
        if num_to_mask == 0:
            return champion_list
        mask_indices = np.random.choice(
            len(champion_list), size=num_to_mask, replace=False
        )
        return [
            self.unknown_champion_id if i in mask_indices else ch_id
            for i, ch_id in enumerate(champion_list)
        ]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = {}

        # Process champion IDs with masking if enabled
        try:
            champion_ids = self._get_champion_ids(row)
            # First encode the original champion IDs
            encoded_champs = self.champion_id_encoder.transform(champion_ids)

            # Then apply masking to the encoded values if needed(the unkown champion id is already encoded)
            if (
                self.masking_function is not None
                and self.unknown_champion_id is not None
            ):
                encoded_champs = self._mask_champions(
                    encoded_champs.tolist(), self.masking_function()
                )

            features["champion_ids"] = torch.tensor(encoded_champs, dtype=torch.long)
        except Exception as e:
            print(f"ERROR processing champion IDs for row {idx}: {e}")
            print(f"Row data: {row}")
            raise

        # Process patch
        features["patch"] = torch.tensor(
            self.patch_mapping[get_patch_from_raw_data(row)], dtype=torch.long
        )

        # We fine tune from the trained ranked_queue_index
        features["queue_type"] = torch.tensor(PRO_QUEUE_INDEX, dtype=torch.long)
        features["elo"] = torch.tensor(
            0.0, dtype=torch.long
        )  # 0 is highest skill level

        # Calculate and normalize task values
        labels = {}

        # Win prediction
        win_prediction = row["team_100_win"]
        if self.use_label_smoothing:
            win_prediction = (
                self.smooth_low if win_prediction == 0 else self.smooth_high
            )
        labels["win_prediction"] = torch.tensor(win_prediction, dtype=torch.float32)

        # Game duration bucketed win predictions
        # gamdeDuration is not normalized in pro dataset
        game_duration_minutes = row["gameDuration"] / 60
        duration_buckets = {
            "0_25": game_duration_minutes < 25,
            "25_30": 25 <= game_duration_minutes < 30,
            "30_35": 30 <= game_duration_minutes < 35,
            "35_inf": game_duration_minutes >= 35,
        }

        for bucket, condition in duration_buckets.items():
            task_name = f"win_prediction_{bucket}"
            if condition:
                labels[task_name] = torch.tensor(win_prediction, dtype=torch.float32)
            else:
                labels[task_name] = torch.tensor(float("nan"), dtype=torch.float32)

        # Gold values at 15 minutes (normalize using task stats)
        for position in POSITIONS:
            for team_id in TEAMS:
                col = f"team_{team_id}_{position}_totalGold_at_900000"
                task_name = col
                gold_value = row[col]

                # Normalize using task statistics
                if self.task_stds[task_name] != 0:
                    normalized_gold = (
                        gold_value - self.task_means[task_name]
                    ) / self.task_stds[task_name]
                else:
                    normalized_gold = gold_value - self.task_means[task_name]

                labels[task_name] = torch.tensor(normalized_gold, dtype=torch.float32)

        return features, labels

    def _get_champion_ids(self, row) -> List[int]:
        """Extract champion IDs from a row of the DataFrame

        Args:
            row: A row from the DataFrame containing champion_ids column

        Returns:
            List of 10 champion IDs (integers)

        Raises:
            ValueError: If champion_ids column is missing or malformed
        """
        if "champion_ids" not in row:
            raise ValueError("Row is missing champion_ids column")

        # it's a numpy array, so we need to convert it to a list
        champion_ids = row["champion_ids"].tolist()
        if not isinstance(champion_ids, list) or len(champion_ids) != 10:
            raise ValueError(
                f"champion_ids must be a list of 10 integers, got: {champion_ids}"
            )

        return champion_ids


def filter_pro_games(pro_games_df: pd.DataFrame, patch_mapping: Dict[str, int]):
    """
    Filter professional games for compatibility with model training data

    Args:
        pro_games_df: DataFrame with professional games
        patch_mapping: Patch mapping dictionary from the original model

    Returns:
        Filtered DataFrame, a list of patches, and counts of filtered games
    """
    original_count = len(pro_games_df)

    # Load champion encoder to check valid champion IDs
    with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
        champion_id_mapping = pickle.load(f)["mapping"]
    valid_champion_ids = set(str(id) for id in champion_id_mapping.classes_)

    # Convert patch mapping keys to version strings for better readability
    patches = sorted(patch_mapping.keys())

    # Filter pro games to include only compatible patches
    patch_filtered_count = 0
    champion_filtered_count = 0
    compatible_games = []

    for _, row in pro_games_df.iterrows():
        patch_str = get_patch_from_raw_data(row)
        # Check patch compatibility
        patch_compatible = patch_str in patches

        # Check champion IDs validity
        champion_ids = [str(champ_id) for champ_id in row["champion_ids"].tolist()]
        champ_compatible = (
            isinstance(champion_ids, list)
            and len(champion_ids) == 10
            and all(champ_id in valid_champion_ids for champ_id in champion_ids)
        )

        if patch_compatible and champ_compatible:
            compatible_games.append(True)
        else:
            compatible_games.append(False)
            if not patch_compatible:
                patch_filtered_count += 1
            if not champ_compatible:
                champion_filtered_count += 1

    pro_games_df = pro_games_df[compatible_games].reset_index(drop=True)

    return (
        pro_games_df,
        patches,
        {
            "original_count": original_count,
            "incompatible_patches": patch_filtered_count,
            "incompatible_champions": champion_filtered_count,
            "remaining_count": len(pro_games_df),
        },
    )


def pro_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Custom collate function for the ProMatchDataset

    Args:
        batch: List of tuples, where each tuple contains:
            - features dict: mapping feature names to individual tensors
            - labels dict: mapping label names to individual tensors

    Returns:
        Tuple of (features_batch, labels_batch), where each is a dict mapping names to batched tensors
    """
    features_batch = {}
    labels_batch = {}

    # Process each item in the batch
    for features, labels in batch:
        # Collect features
        for k, v in features.items():
            if k not in features_batch:
                features_batch[k] = []
            features_batch[k].append(v)

        # Collect labels
        for k, v in labels.items():
            if k not in labels_batch:
                labels_batch[k] = []
            labels_batch[k].append(v)

    # Convert lists to tensors
    features_batch = {
        k: torch.stack(v) if v[0].dim() > 0 else torch.tensor(v)
        for k, v in features_batch.items()
    }
    labels_batch = {
        k: torch.stack(v) if v[0].dim() > 0 else torch.tensor(v)
        for k, v in labels_batch.items()
    }

    return features_batch, labels_batch


class MixedDataLoader:
    """Custom data loader that combines original and fine-tuning data"""

    def __init__(
        self,
        original_loader: DataLoader,
        finetune_loader: DataLoader,
    ):
        self.original_loader = original_loader
        self.finetune_loader = finetune_loader
        self.original_iter = iter(original_loader)
        self.length = len(finetune_loader)

        # Load task statistics once during initialization
        with open(TASK_STATS_PATH, "rb") as f:
            task_stats = pickle.load(f)
            self.game_duration_mean = task_stats["means"]["gameDuration"]
            self.game_duration_std = task_stats["stds"]["gameDuration"]

    def __len__(self):
        return self.length

    def __iter__(self):
        # Create fresh iterators at the start of each epoch
        # This is not done for original loader to have it iterated over internally
        self.finetune_iter = iter(self.finetune_loader)
        return self

    def __next__(self):
        try:
            # Get next fine-tuning batch
            finetune_features, finetune_labels = next(self.finetune_iter)

            try:
                original_features, original_labels = next(self.original_iter)
            except StopIteration:
                # Reset original data iterator if exhausted
                print("Resetting original data iterator")
                self.original_iter = iter(self.original_loader)
                original_features, original_labels = next(self.original_iter)

            # Combine batches
            combined_features = {}
            combined_labels = {}

            for key in finetune_features:
                combined_features[key] = torch.cat(
                    [finetune_features[key], original_features[key]]
                )

            # Only include fine-tune tasks in labels
            for key in FINE_TUNE_TASKS:
                if not key.startswith("win_prediction_"):
                    # For non-win prediction tasks, just concatenate as before
                    combined_labels[key] = torch.cat(
                        [finetune_labels[key], original_labels[key]]
                    )
                else:
                    # For win_prediction bucketed tasks
                    if key == "win_prediction":
                        # The main win_prediction task is handled normally
                        combined_labels[key] = torch.cat(
                            [finetune_labels[key], original_labels["win_prediction"]]
                        )
                    else:
                        # Get bucket range from task name
                        bucket = key.split("win_prediction_")[1]

                        # Denormalize game duration for original data using stored stats
                        orig_duration_minutes = (
                            original_labels["gameDuration"] * self.game_duration_std
                            + self.game_duration_mean
                        ) / 60

                        # TODO: could add time buckets here for both? instead of also having the logic in the pro dataset?
                        # Create duration mask based on bucket
                        if bucket == "0_25":
                            duration_mask = orig_duration_minutes < 25
                        elif bucket == "25_30":
                            duration_mask = (orig_duration_minutes >= 25) & (
                                orig_duration_minutes < 30
                            )
                        elif bucket == "30_35":
                            duration_mask = (orig_duration_minutes >= 30) & (
                                orig_duration_minutes < 35
                            )
                        elif bucket == "35_inf":
                            duration_mask = orig_duration_minutes >= 35

                        # Create masked labels: use win_prediction where duration matches, NaN otherwise
                        orig_masked_labels = torch.full_like(
                            original_labels["win_prediction"],
                            float("nan"),
                            dtype=torch.float32,
                        )
                        orig_masked_labels[duration_mask] = original_labels[
                            "win_prediction"
                        ][duration_mask]

                        # Combine finetune and masked original labels
                        combined_labels[key] = torch.cat(
                            [finetune_labels[key], orig_masked_labels]
                        )

            return combined_features, combined_labels

        except StopIteration:
            # End epoch when fine-tuning data is exhausted
            raise StopIteration


def split_data_by_teams(
    df: pd.DataFrame, val_split: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Split the data by randomly selecting teams for validation until reaching desired split ratio.

    Args:
        df: DataFrame containing the pro games data
        val_split: Desired fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        val_teams: Set of team names selected for validation
    """
    random.seed(seed)

    # Get unique team names
    all_teams = set(df["blueTeamName"].unique()) | set(df["redTeamName"].unique())
    all_teams = list(all_teams)

    val_teams = set()
    val_indices = set()

    # Keep adding teams to validation set until we reach desired split
    while len(val_indices) < len(df) * val_split:
        if not all_teams:
            break

        # Randomly select a team
        team = random.choice(all_teams)
        all_teams.remove(team)
        val_teams.add(team)

        # Find all games where this team played (either blue or red)
        team_matches = df.index[
            (df["blueTeamName"] == team) | (df["redTeamName"] == team)
        ].tolist()
        val_indices.update(team_matches)

    # Create train/val masks
    val_mask = df.index.isin(val_indices)
    train_mask = ~val_mask

    # Split the data
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    print(f"Selected {len(val_teams)} teams for validation:")
    print(f"Validation teams: {sorted(val_teams)}")
    print(f"Training data: {len(train_df)} games")
    print(f"Validation data: {len(val_df)} games")

    return train_df, val_df, val_teams


def split_data_by_champions(
    df: pd.DataFrame, val_split: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int]]:
    """
    Split the data by randomly selecting champions for validation until reaching desired split ratio.
    A game goes to validation if it contains any of the validation champions in any role.

    Args:
        df: DataFrame containing the pro games data
        val_split: Desired fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        val_champions: Set of champion IDs selected for validation
    """
    random.seed(seed)

    # Get all unique champion IDs from all games
    all_champions = set()
    for champ_list in df["champion_ids"]:
        all_champions.update(champ_list)
    all_champions = list(all_champions)  # Convert to list for random selection

    print(f"Total unique champions in dataset: {len(all_champions)}")

    val_champions = set()
    val_indices = set()

    # Keep adding champions to validation set until we reach desired split
    while len(val_indices) < len(df) * val_split:
        if not all_champions:
            break

        # Randomly select a champion
        champion = random.choice(all_champions)
        all_champions.remove(champion)
        val_champions.add(champion)

        # Find all games where this champion appears in any role
        champion_matches = df.index[
            df["champion_ids"].apply(lambda x: champion in x)
        ].tolist()
        val_indices.update(champion_matches)

    # Create train/val masks
    val_mask = df.index.isin(val_indices)
    train_mask = ~val_mask

    # Split the data
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    # Calculate champion appearance statistics
    train_champ_counts = {}
    val_champ_counts = {}

    for champ_list in train_df["champion_ids"]:
        for champ in champ_list:
            train_champ_counts[champ] = train_champ_counts.get(champ, 0) + 1

    for champ_list in val_df["champion_ids"]:
        for champ in champ_list:
            val_champ_counts[champ] = val_champ_counts.get(champ, 0) + 1

    print("\nChampion-based split statistics:")
    print(f"Selected {len(val_champions)} champions for validation")
    print(f"Training data: {len(train_df)} games")
    print(f"Validation data: {len(val_df)} games")

    print("\nValidation champion appearances:")
    for champ in sorted(val_champions):
        train_count = train_champ_counts.get(champ, 0)
        val_count = val_champ_counts.get(champ, 0)
        print(
            f"Champion {champ}: {val_count} validation games, {train_count} training games"
        )

    # Verify complete isolation
    train_champions = set()
    for champ_list in train_df["champion_ids"]:
        train_champions.update(champ_list)

    overlap = train_champions & val_champions
    if overlap:
        print("\nWARNING: Found validation champions in training set!")
        print(f"Overlapping champions: {sorted(overlap)}")

    return train_df, val_df, val_champions


def finetune_masking_function():
    if np.random.rand() < 0.25:
        return np.random.randint(1, 11)
    else:
        return 0


def create_dataloaders(
    pro_games_df,
    patch_mapping,
    config,
    split_strategy: str = "random",
):
    """Create train and validation dataloaders for fine-tuning

    Args:
        pro_games_df: DataFrame containing pro games data
        patch_mapping: Dictionary mapping patch strings to indices
        config: Configuration object
        split_strategy: One of "random", "team", "patch", or "champion"
    """
    # First calculate team winrates
    team_stats = {}

    # Process all games to calculate winrates
    for _, row in pro_games_df.iterrows():
        blue_team = row["blueTeamName"]
        red_team = row["redTeamName"]
        blue_win = row["team_100_win"]

        # Update blue team stats
        if blue_team not in team_stats:
            team_stats[blue_team] = {"wins": 0, "total_games": 0}
        team_stats[blue_team]["wins"] += blue_win
        team_stats[blue_team]["total_games"] += 1

        # Update red team stats
        if red_team not in team_stats:
            team_stats[red_team] = {"wins": 0, "total_games": 0}
        team_stats[red_team]["wins"] += 1 - blue_win
        team_stats[red_team]["total_games"] += 1

    balanced_teams = set()
    min_games = 5
    min_winrate = 0.1
    max_winrate = 0.9

    for team, stats in team_stats.items():
        if stats["total_games"] >= min_games:
            winrate = stats["wins"] / stats["total_games"]
            if min_winrate <= winrate <= max_winrate:
                balanced_teams.add(team)

    print(
        f"\nFound {len(balanced_teams)} teams with winrates between {min_winrate*100}% and {max_winrate*100}%"
    )

    # Perform the regular split based on strategy
    if split_strategy == "team":
        train_df, val_df, val_teams = split_data_by_teams(
            pro_games_df, val_split=config.val_split
        )
    elif split_strategy == "patch":
        train_df, val_df, val_patches = split_data_by_patches(
            pro_games_df, val_split=config.val_split
        )
    elif split_strategy == "champion":
        train_df, val_df, val_champions = split_data_by_champions(
            pro_games_df, val_split=config.val_split
        )
    else:  # random split
        train_df, val_df = split_data_randomly(pro_games_df, val_split=config.val_split)

    # Filter training set to only include games between balanced teams
    original_train_size = len(train_df)
    train_df = train_df[
        train_df.apply(
            lambda row: (row["blueTeamName"] in balanced_teams)
            and (row["redTeamName"] in balanced_teams),
            axis=1,
        )
    ].reset_index(drop=True)

    print(f"\nTraining set filtering:")
    print(f"Original training set size: {original_train_size} games")
    print(f"Filtered training set size: {len(train_df)} games")
    print(f"Removed {original_train_size - len(train_df)} games with unbalanced teams")

    # Get num_champions and unknown_champion_id using the utility function
    # TODO: this could be inside the dataset class
    _, unknown_champion_id = get_num_champions()

    # Create masking strategy for both datasets
    masking_strategy = MASKING_STRATEGIES[
        "strategic"
    ]()  # Use strategic masking by default

    # Create pro datasets with already split data
    train_pro_dataset = ProMatchDataset(
        pro_games_df=train_df,
        patch_mapping=patch_mapping,
        use_label_smoothing=config.use_label_smoothing,
        smooth_low=config.smooth_low,
        smooth_high=config.smooth_high,
        masking_function=finetune_masking_function,
        unknown_champion_id=unknown_champion_id,
    )
    print(f"Created train dataset with {len(train_pro_dataset)} pro games")

    val_pro_dataset = ProMatchDataset(
        pro_games_df=val_df,
        patch_mapping=patch_mapping,
        use_label_smoothing=False,
    )
    print(f"Created test dataset with {len(val_pro_dataset)} pro games")

    # Create pro dataloaders
    train_pro_loader = DataLoader(
        train_pro_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pro_collate_fn,
    )

    val_pro_loader = DataLoader(
        val_pro_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pro_collate_fn,
    )

    # Check if we should use original data
    if config.original_batch_size > 0:
        # Create original dataset with the SAME masking strategy
        train_original_dataset = MatchDataset(
            train_or_test="train",
            dataset_fraction=1.0,  # Use full dataset for training
            masking_function=masking_strategy,  # Add masking to original dataset
            unknown_champion_id=unknown_champion_id,  # Add unknown champion ID
        )
        val_original_dataset = MatchDataset(
            train_or_test="test",
            dataset_fraction=0.01,
            masking_function=masking_strategy,  # Add masking to validation set too
            unknown_champion_id=unknown_champion_id,
        )

        # Create original dataloaders
        train_original_loader = DataLoader(
            train_original_dataset,
            batch_size=config.original_batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            **dataloader_config,
        )

        val_original_loader = DataLoader(
            val_original_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            **dataloader_config,
        )

        # Create mixed training loader
        train_loader = MixedDataLoader(
            train_original_loader,
            train_pro_loader,
        )
    else:
        # If original_batch_size is 0, use only pro data
        print("Using only pro data for training (original_batch_size = 0)")
        train_loader = train_pro_loader
        val_original_loader = None

    return train_loader, val_pro_loader, val_original_loader


def unfreeze_layer_group(model: Model, frozen_layers: int) -> int:
    """
    Unfreeze the next group of layers in the model.
    Returns the new count of frozen layer groups.
    """
    mlp_layers = list(model.mlp_layers)
    # Keep at least one layer group frozen
    if frozen_layers <= 1:
        print("Cannot unfreeze more layers: minimum of 1 frozen layer group required")
        return 1

    # Calculate which layer group to unfreeze (each group is 4 layers)
    base_index = (frozen_layers - 1) * 4

    # Unfreeze all four layers in the group
    for i in range(4):
        layer_idx = base_index + i
        mlp_layers[layer_idx].requires_grad_(True)
        print(f"Unfrozing layer at index {layer_idx}: {mlp_layers[layer_idx]}")

    # Add this line to ensure BatchNorm layers stay frozen
    model.apply(freeze_bn)

    return frozen_layers - 1


def freeze_bn(module):
    """Helper function to freeze batch norm layers"""
    if isinstance(module, nn.BatchNorm1d):
        module.eval()  # Set to evaluation mode
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(False)


def fine_tune_model(
    pretrained_model_path: str,
    pro_games_df: pd.DataFrame,
    finetune_config: FineTuningConfig,
    output_model_path: str,
    run_name: Optional[str] = None,
    split_strategy: str = "random",
):
    """Fine-tune a pre-trained model on professional game data"""
    device = get_best_device()
    print(f"Using device: {device}")

    # Load patch mapping
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]

    # Initialize unfreezing state
    frozen_layers = finetune_config.initial_frozen_layers
    next_unfreeze_idx = 0  # Track which epoch milestone we're waiting for

    main_model_config = TrainingConfig()
    # Initialize the model
    model = Model(
        config=main_model_config,
        hidden_dims=main_model_config.hidden_dims,
        dropout=finetune_config.dropout,
    )

    model = load_model_state_dict(model, device, path=pretrained_model_path)

    # Track best model and metrics
    best_metric = float("inf")
    best_model_state = None

    # Initially freeze ALL embeddings including queue_type
    print("Initially freezing all embedding layers...")
    model.patch_embedding.requires_grad_(False)
    model.champion_patch_embedding.requires_grad_(False)
    model.champion_embedding.requires_grad_(False)
    for name, embedding in model.embeddings.items():
        # We don't freeze queue_type, to include original data but let model differentiate between pro and original data
        if name != "queue_type":
            embedding.requires_grad_(False)

    # Freeze early MLP layers
    print(f"Freezing first {frozen_layers} MLP layer groups...")
    mlp_layers = list(model.mlp_layers)
    layers_to_freeze = frozen_layers * 4
    for layer in mlp_layers[:layers_to_freeze]:
        print(f"Freezing layer: {layer}")
        layer.requires_grad_(False)

    # Add this line to freeze batch norm layers
    model.apply(freeze_bn)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})"
    )

    # Initialize wandb
    if finetune_config.log_wandb:
        wandb.init(
            project="draftking-pro-finetune",
            name=run_name or f"finetune-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                **finetune_config.to_dict(),
                "pretrained_model": pretrained_model_path,
                "pro_games_count": len(pro_games_df),
                "trainable_params": trainable_params,
                "total_params": total_params,
            },
        )
        wandb.watch(model, log_freq=100)

    # Define loss functions for each task type
    criterion = {
        TaskType.BINARY_CLASSIFICATION: nn.BCEWithLogitsLoss(reduction="none"),
        TaskType.REGRESSION: nn.MSELoss(reduction="none"),
    }

    # Get task definitions and weights
    enabled_tasks = FINE_TUNE_TASKS
    task_names = list(enabled_tasks.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in enabled_tasks.values()], device=device
    )

    # Initialize optimizer with initially trainable parameters
    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, finetune_config.weight_decay),
        lr=finetune_config.learning_rate,
    )

    train_loader, val_pro_loader, val_original_loader = create_dataloaders(
        pro_games_df,
        patch_mapping,
        finetune_config,
        split_strategy=split_strategy,
    )

    # Store the validation DataFrame for later use
    val_df = val_pro_loader.dataset.df  # Get the validation DataFrame from the dataset

    # Check if we're using only pro data
    using_only_pro_data = finetune_config.original_batch_size == 0

    for epoch in range(finetune_config.num_epochs):
        epoch_start = time.time()

        # Check if we should unfreeze the next layer group
        if (
            next_unfreeze_idx < len(finetune_config.epoch_to_unfreeze)
            and epoch >= finetune_config.epoch_to_unfreeze[next_unfreeze_idx]
        ):

            frozen_layers = unfreeze_layer_group(model, frozen_layers)
            print(
                f"\nEpoch {epoch}: Unfroze layer group. Remaining frozen groups: {frozen_layers}"
            )

            # Reinitialize optimizer with newly unfrozen parameters
            optimizer = optim.AdamW(
                get_optimizer_grouped_parameters(model, finetune_config.weight_decay),
                lr=finetune_config.learning_rate,
            )

            next_unfreeze_idx += 1

        # Training
        model.train()
        # Re-freeze BatchNorm layers after model.train()
        model.apply(freeze_bn)

        train_losses = {task: 0.0 for task in task_names}
        pro_train_losses = {task: 0.0 for task in task_names}
        original_train_losses = {task: 0.0 for task in task_names}
        pro_train_counts = {task: 0 for task in task_names}
        original_train_counts = {task: 0 for task in task_names}
        train_steps = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{finetune_config.num_epochs}"
        )
        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(features)

            # Calculate losses for each task
            batch_losses = []
            pro_losses = {task: 0.0 for task in task_names}
            original_losses = {task: 0.0 for task in task_names}

            for task_name, task_def in enabled_tasks.items():
                task_criterion = criterion[task_def.task_type]

                # Create mask for non-NaN values in both predictions and labels
                valid_mask = ~torch.isnan(labels[task_name])
                if valid_mask.any():
                    # Only compute loss on non-NaN values
                    valid_outputs = outputs[task_name][valid_mask]
                    valid_labels = labels[task_name][valid_mask]

                    # Additional check for NaN in outputs
                    output_valid_mask = ~torch.isnan(valid_outputs)
                    if output_valid_mask.any():
                        task_loss = task_criterion(
                            valid_outputs[output_valid_mask],
                            valid_labels[output_valid_mask],
                        )

                        if using_only_pro_data:
                            # If using only pro data, all samples are pro samples
                            pro_loss = task_loss.mean()
                            weighted_loss = pro_loss * finetune_config.pro_loss_weight
                            pro_losses[task_name] = pro_loss.item()
                            pro_train_losses[task_name] += pro_loss.item()
                            pro_train_counts[task_name] += 1
                            batch_losses.append(weighted_loss)
                        else:
                            # Create pro/original mask before applying output_valid_mask
                            pro_orig_mask = (
                                torch.arange(len(valid_mask), device=device)
                                < finetune_config.batch_size
                            )
                            # Keep track of which valid samples are from pro/original data
                            valid_pro_mask = pro_orig_mask[valid_mask]

                            # Update pro mask to account for output_valid_mask
                            final_pro_mask = valid_pro_mask[output_valid_mask]

                            # Apply separate weights to pro and original losses
                            weighted_loss_sum = 0.0

                            # Apply pro weight to pro samples
                            if final_pro_mask.any():
                                pro_loss = task_loss[final_pro_mask].mean()
                                weighted_pro_loss = (
                                    pro_loss * finetune_config.pro_loss_weight
                                )
                                pro_losses[task_name] = pro_loss.item()
                                pro_train_losses[task_name] += pro_loss.item()
                                pro_train_counts[task_name] += 1
                                # Add weighted pro loss
                                weighted_loss_sum += weighted_pro_loss

                            # Apply original weight to original samples
                            if (~final_pro_mask).any():
                                original_loss = task_loss[~final_pro_mask].mean()
                                weighted_original_loss = (
                                    original_loss * finetune_config.original_loss_weight
                                )
                                original_losses[task_name] = original_loss.item()
                                original_train_losses[task_name] += original_loss.item()
                                original_train_counts[task_name] += 1
                                # Add weighted original loss
                                weighted_loss_sum += weighted_original_loss

                            # Use weighted loss - now a scalar value
                            batch_losses.append(weighted_loss_sum)
                    else:
                        print(f"No valid values for task {task_name}")
                        batch_losses.append(torch.tensor(0.0, device=device))
                else:
                    print(f"No valid values for task {task_name}")
                    batch_losses.append(torch.tensor(0.0, device=device))

            # Combine losses with task weights
            batch_losses = torch.stack(batch_losses)

            # Debug information for weighted loss
            if torch.isnan(batch_losses).any():
                print("\nNaN detected in batch losses!")
                for task_name, loss in zip(task_names, batch_losses):
                    print(f"{task_name}: {loss.item()}")
                print(f"Task weights: {task_weights}")

            total_loss = (batch_losses * task_weights).sum()

            # Log per-batch losses to wandb
            if finetune_config.log_wandb:
                log_dict = {
                    f"train_batch_pro_loss_{task}": loss
                    for task, loss in pro_losses.items()
                    if loss is not None
                }
                if not using_only_pro_data:
                    log_dict.update(
                        {
                            f"train_batch_original_loss_{task}": loss
                            for task, loss in original_losses.items()
                            if loss is not None
                        }
                    )
                wandb.log(log_dict)

            # Backward pass
            total_loss.backward()

            clip_grad_norm_(model.parameters(), finetune_config.max_grad_norm)

            optimizer.step()
            train_steps += 1

        # Calculate average losses and metrics
        avg_train_losses = {
            task: loss / train_steps for task, loss in train_losses.items()
        }

        # Calculate average pro and original losses
        avg_pro_train_losses = {
            task: loss / count if count > 0 else float("nan")
            for task, (loss, count) in zip(
                pro_losses.keys(),
                zip(pro_losses.values(), pro_train_counts.values()),
            )
        }

        avg_original_train_losses = {
            task: loss / count if count > 0 else float("nan")
            for task, (loss, count) in zip(
                original_losses.keys(),
                zip(original_losses.values(), original_train_counts.values()),
            )
        }

        # Validation
        model.eval()

        # Save periodic checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0 and not finetune_config.debug:
            # Add run name to periodic checkpoints if available
            run_suffix = f"_{run_name}" if run_name else ""
            checkpoint_path = output_model_path.replace(
                ".pth", f"{run_suffix}_epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Only run validation on specified intervals
        if (epoch + 1) % finetune_config.validation_interval == 0:
            # Regular validation, with no masking
            val_metrics = validate(
                model,
                val_pro_loader,
                device,
            )
            val_loss = val_metrics["val_pro_avg_loss"]

            # Save best model based on validation loss
            if val_loss < best_metric:
                best_metric = val_loss
                if not finetune_config.debug:
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_model_path = output_model_path.replace(".pth", "_best.pth")
                    torch.save(best_model_state, best_model_path)
                    print(
                        f"New best model saved with validation loss: {best_metric:.4f}"
                    )

            # Masked validation with different masking levels - now using validation data only
            masked_metrics = validate_with_masking_levels(
                model=model,
                val_df=val_df,  # Pass validation DataFrame instead of full dataset
                patch_mapping=patch_mapping,
                config=finetune_config,
                device=device,
                epoch=epoch,
            )

            # Log all metrics
            if finetune_config.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        **val_metrics,
                        **masked_metrics,
                    }
                )

        epoch_time = time.time() - epoch_start

        # Log metrics
        if finetune_config.log_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                **{
                    f"train_loss_{task}": loss
                    for task, loss in avg_train_losses.items()
                },
                **{
                    f"train_pro_loss_{task}": loss
                    for task, loss in avg_pro_train_losses.items()
                },
            }
            if not using_only_pro_data:
                log_dict.update(
                    {
                        f"train_original_loss_{task}": loss
                        for task, loss in avg_original_train_losses.items()
                    }
                )
            wandb.log(log_dict)

        # Print progress
        print(
            f"Epoch {epoch+1}/{finetune_config.num_epochs} completed in {epoch_time:.2f}s"
        )

    # Save final model
    if not finetune_config.debug:
        torch.save(model.state_dict(), output_model_path)
        print(f"Final model saved to {output_model_path}")

        # Also save best model if it wasn't the final one
        if best_model_state is not None:
            best_model_path = output_model_path.replace(".pth", "_best.pth")
            torch.save(best_model_state, best_model_path)
            print(f"Best model saved with validation loss: {best_metric:.4f}")

    if finetune_config.log_wandb:
        wandb.finish()

    return


def validate(
    model: Model,
    val_pro_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation on both pro and original data"""
    model.eval()
    enabled_tasks = FINE_TUNE_TASKS

    # Initialize accumulators for both datasets
    pro_accumulators = {
        task_name: torch.zeros(2, device=device) for task_name in enabled_tasks.keys()
    }

    # Add accuracy accumulator for win_prediction
    pro_win_accuracy = torch.zeros(2, device=device)

    with torch.no_grad():
        # Validate on pro data
        pro_metrics = validate_loader(
            model=model,
            loader=val_pro_loader,
            accumulators=pro_accumulators,
            win_accuracy=pro_win_accuracy,
            device=device,
            prefix="val_pro",
        )

    return pro_metrics


def validate_loader(
    model: Model,
    loader: DataLoader,
    accumulators: Dict[str, torch.Tensor],
    win_accuracy: torch.Tensor,
    device: torch.device,
    prefix: str,
) -> Dict[str, float]:
    """Validate model on a single loader"""
    total_steps = 0
    weighted_task_losses = torch.zeros(len(FINE_TUNE_TASKS), device=device)
    task_weights = torch.tensor(
        [task_def.weight for task_def in FINE_TUNE_TASKS.values()], device=device
    )

    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}
        labels = {k: v.to(device) for k, v in labels.items()}
        batch_size = next(iter(labels.values())).size(0)

        outputs = model(features)

        # Calculate task losses
        for task_idx, (task_name, task_def) in enumerate(FINE_TUNE_TASKS.items()):
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs[task_name], labels[task_name], reduction="none"
                )
            else:  # REGRESSION
                loss = nn.functional.mse_loss(
                    outputs[task_name], labels[task_name], reduction="none"
                )

            # Update accumulators
            valid_mask = ~torch.isnan(labels[task_name])
            if valid_mask.any():
                loss_sum = loss[valid_mask].sum()
                accumulators[task_name][0] += loss_sum
                accumulators[task_name][1] += valid_mask.sum()
                # Add to weighted task losses
                weighted_task_losses[task_idx] += loss[valid_mask].mean().item()

        # Calculate win prediction accuracy
        if "win_prediction" in FINE_TUNE_TASKS:
            win_probs = torch.sigmoid(outputs["win_prediction"])
            predictions = (win_probs > 0.5).float()
            correct = (predictions == labels["win_prediction"]).sum()
            win_accuracy[0] += correct
            win_accuracy[1] += batch_size

        total_steps += 1

    # Calculate metrics
    metrics = {}

    # Add individual task losses
    for task_name, accumulator in accumulators.items():
        if accumulator[1] > 0:  # Only calculate if we have valid samples
            metrics[f"{prefix}_{task_name}_loss"] = (
                accumulator[0] / accumulator[1]
            ).item()

    # Add win prediction accuracy if available
    if win_accuracy[1] > 0:
        metrics[f"{prefix}_win_prediction_accuracy"] = (
            win_accuracy[0] / win_accuracy[1]
        ).item()

    # Calculate weighted average loss across all tasks
    if total_steps > 0:
        avg_weighted_task_losses = weighted_task_losses / total_steps
        metrics[f"{prefix}_avg_loss"] = (
            (avg_weighted_task_losses * task_weights).sum().item()
        )
    else:
        metrics[f"{prefix}_avg_loss"] = float("inf")

    return metrics


def validate_with_masking_levels(
    model: Model,
    val_df: pd.DataFrame,
    patch_mapping: Dict[str, int],
    config: FineTuningConfig,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate the model with different numbers of masked champions on validation data only
    Returns a dictionary of metrics for each masking level
    """
    all_metrics = {}

    # Get num_champions and unknown_champion_id using the utility function
    _, unknown_champion_id = get_num_champions()

    # Test with different numbers of masked champions
    for num_masked in range(1, 11):  # 1 to 10 masked champions
        np.random.seed(42 + num_masked)

        # Create dataset with specific number of masked champions
        val_masked_dataset = ProMatchDataset(
            pro_games_df=val_df,
            patch_mapping=patch_mapping,
            use_label_smoothing=False,
            masking_function=lambda: num_masked,  # Always mask exactly num_masked champions
            unknown_champion_id=unknown_champion_id,
        )

        val_masked_loader = DataLoader(
            val_masked_dataset,
            batch_size=config.batch_size,
            shuffle=False,  # Ensure no shuffling
            collate_fn=pro_collate_fn,
        )

        # Run validation
        metrics = validate(
            model=model,
            val_pro_loader=val_masked_loader,
            device=device,
        )

        # Add prefix to metrics
        masked_metrics = {f"val_masked_{num_masked}_{k}": v for k, v in metrics.items()}
        all_metrics.update(masked_metrics)

        # Print some key metrics
        if "win_prediction" in metrics:
            print(
                f"Win prediction loss with {num_masked} masked: {metrics['win_prediction']:.4f}"
            )
        if "win_prediction_accuracy" in metrics:
            print(
                f"Win prediction accuracy with {num_masked} masked: {metrics['win_prediction_accuracy']:.4f}"
            )

    return all_metrics


def main():
    """Main function to run fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Fine-tune model on professional game data"
    )

    # Update split strategy choices
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["random", "team", "patch", "champion"],
        default="random",
        help="Strategy to split data into train/validation sets",
    )

    # Keep only essential path-related arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pre-trained model to fine-tune",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the fine-tuned model (default: model_path with _pro_finetuned suffix)",
    )

    parser.add_argument(
        "--pro-data-path",
        type=str,
        default=RAW_PRO_GAMES_FILE,
        help="Path to the professional games parquet file",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the wandb run",
    )

    args = parser.parse_args()

    # Set up output path if not specified
    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path)
        name_parts = model_name.rsplit(".", 1)
        args.output_path = os.path.join(
            model_dir,
            f"{name_parts[0]}_pro_finetuned.{name_parts[1] if len(name_parts) > 1 else 'pth'}",
        )

    # Load pro game data
    pro_games_df = pd.read_parquet(args.pro_data_path)
    print(f"Loaded {len(pro_games_df)} professional games from {args.pro_data_path}")

    # Load patch mapping to filter for compatible patches
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]

    # Filter games
    pro_games_df, patches, filter_stats = filter_pro_games(pro_games_df, patch_mapping)

    print(f"Original model was trained on {len(patches)} patches: {', '.join(patches)}")
    print(
        f"Filtered out {filter_stats['incompatible_patches']} games with incompatible patches"
    )
    print(f"Remaining pro games for fine-tuning: {filter_stats['remaining_count']}")

    if len(pro_games_df) == 0:
        print("Error: No compatible pro games found. Cannot proceed with fine-tuning.")
        return

    # Create configuration for fine-tuning - edit this object directly in the code
    config = FineTuningConfig()

    print(f"Fine-tuning configuration:")
    print(config)

    # Set random seeds for reproducibility
    set_random_seeds()

    # Start fine-tuning
    fine_tune_model(
        pretrained_model_path=args.model_path,
        pro_games_df=pro_games_df,
        finetune_config=config,
        output_model_path=args.output_path,
        run_name=args.run_name,
        split_strategy=args.split_strategy,
    )


if __name__ == "__main__":
    main()
