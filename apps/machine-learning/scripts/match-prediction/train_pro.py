# scripts/match-prediction/train_pro.py
# Finetunes the model on professional data
# (Doesn't work well, might be removed soon, it is a research area)

import os
import argparse
import torch
import wandb
import pickle
import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

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

# Attempting to finetune on multiple tasks, could perhaps make the model "understand" the domain adaptations better
# (for example, pro games have less drastic gold leads and less kills)
FINE_TUNE_TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=1,
    ),
}

FINE_TUNE_TASKS_LAST_EPOCHS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=1,
    ),
}


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
        # Fine-tuning hyperparameters - edit these directly instead of using command line flags
        self.num_epochs = 1000
        # TODO: try even lower? original is 8e-4 right now
        self.learning_rate = 1.6e-5  # Lower learning rate for fine-tuning
        self.weight_decay = 0.05
        self.dropout = 0.5
        self.batch_size = 1024
        self.original_batch_size = 0
        self.val_split = 0.2
        self.max_grad_norm = 1.0
        self.log_wandb = True
        self.save_checkpoints = False

        # New unfreezing parameters
        self.progressive_unfreezing = True  # Enable progressive unfreezing
        self.epochs_per_unfreeze = 400
        self.initial_frozen_layers = 3

        # Data augmentation options
        self.use_team_symmetry = False

        # Label smoothing options
        self.use_label_smoothing = True  # Enable label smoothing
        self.smooth_low = 0.2  # Value to smooth 0 labels to
        self.smooth_high = 0.8  # Value to smooth 1 labels to

        # Loss balancing parameters
        self.pro_loss_weight = 1.0  # Weight multiplier for pro data losses
        self.original_loss_weight = 1.0  # Weight multiplier for original data losses

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def to_dict(self):
        return {key: value for key, value in vars(self).items()}


class ProMatchDataset(Dataset):
    """Dataset for professional matches fine-tuning"""

    def __init__(
        self,
        pro_games_df: pd.DataFrame,
        patch_mapping: Dict[str, int],
        train_or_test: str = "train",
        val_split: float = 0.2,
        use_team_symmetry: bool = True,
        use_label_smoothing: bool = True,
        smooth_low: float = 0.1,
        smooth_high: float = 0.9,
        seed: int = 42,
    ):
        with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
            # TODO: could have a function for this, it is done at many places and not obvious to use "mapping"
            self.champion_id_encoder = pickle.load(f)["mapping"]

        self.patch_mapping = patch_mapping

        # Split data into train/test
        np.random.seed(seed)
        indices = np.random.permutation(len(pro_games_df))
        split_idx = int(len(indices) * (1 - val_split))

        if train_or_test == "train":
            self.df = pro_games_df.iloc[indices[:split_idx]].reset_index(drop=True)
        else:
            self.df = pro_games_df.iloc[indices[split_idx:]].reset_index(drop=True)

        print(f"Created {train_or_test} dataset with {len(self.df)} pro games")

        self.use_team_symmetry = use_team_symmetry

        # Label smoothing parameters
        self.use_label_smoothing = use_label_smoothing
        self.smooth_low = smooth_low
        self.smooth_high = smooth_high

        # Load task statistics for normalization
        with open(TASK_STATS_PATH, "rb") as f:
            task_stats = pickle.load(f)
            self.task_means = task_stats["means"]
            self.task_stds = task_stats["stds"]

        # Define task symmetry transformations
        self.task_symmetry = {
            "win_prediction": lambda x: 1.0 - x,
        }

    def __len__(self):
        return len(self.df) * (2 if self.use_team_symmetry else 1)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        # Handle symmetry indexing
        use_symmetry = self.use_team_symmetry and idx >= len(self.df)
        original_idx = idx % len(self.df)
        row = self.df.iloc[original_idx]

        # Extract features
        features = {}

        # Process champion IDs with symmetry handling
        try:
            champion_ids = self._get_champion_ids(row)
            if use_symmetry:
                # Swap blue and red team champions (first 5 with last 5)
                champion_ids = champion_ids[5:] + champion_ids[:5]

            encoded_champs = self.champion_id_encoder.transform(champion_ids)
            features["champion_ids"] = torch.tensor(encoded_champs, dtype=torch.long)
        except Exception as e:
            print(f"ERROR processing champion IDs for row {idx}: {e}")
            print(f"Row data: {row}")
            raise

        # Process patch
        features["patch"] = torch.tensor(
            self.patch_mapping[get_patch_from_raw_data(row)], dtype=torch.long
        )

        features["queue_type"] = torch.tensor(RANKED_QUEUE_INDEX, dtype=torch.long)
        # Add numerical_elo = 0 for pro games (highest skill level)
        # TODO: could have a new elo as well
        features["elo"] = torch.tensor(0.0, dtype=torch.long)

        # Calculate and normalize task values
        labels = {}

        # Win prediction
        win_prediction = row["team_100_win"]
        if use_symmetry:
            win_prediction = self.task_symmetry["win_prediction"](win_prediction)
        if self.use_label_smoothing:
            win_prediction = (
                self.smooth_low if win_prediction == 0 else self.smooth_high
            )
        labels["win_prediction"] = torch.tensor(win_prediction, dtype=torch.float32)

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
        champ_compatible = (
            isinstance(row["champion_ids"].tolist(), list)
            and len(row["champion_ids"].tolist()) == 10
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
                combined_labels[key] = torch.cat(
                    [finetune_labels[key], original_labels[key]]
                )

            return combined_features, combined_labels

        except StopIteration:
            # End epoch when fine-tuning data is exhausted
            raise StopIteration


def create_dataloaders(
    pro_games_df,
    patch_mapping,
    config,
):
    """Create train and validation dataloaders for fine-tuning"""

    # Create pro datasets
    train_pro_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        patch_mapping=patch_mapping,
        train_or_test="train",
        val_split=config.val_split,
        use_team_symmetry=config.use_team_symmetry,
        use_label_smoothing=config.use_label_smoothing,
        smooth_low=config.smooth_low,
        smooth_high=config.smooth_high,
    )

    val_pro_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        patch_mapping=patch_mapping,
        train_or_test="test",
        val_split=config.val_split,
        use_team_symmetry=False,
        use_label_smoothing=False,
    )

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
        # Create original dataset (using only fine-tune tasks)
        # Use full dataset for training but fraction for validation
        train_original_dataset = MatchDataset(
            train_or_test="train", dataset_fraction=1.0  # Use full dataset for training
        )
        val_original_dataset = MatchDataset(train_or_test="test", dataset_fraction=0.01)

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
        # Create a dummy original loader for validation
        val_original_loader = None

    return train_loader, val_pro_loader, val_original_loader


def unfreeze_layer_group(model: Model, frozen_layers: int) -> int:
    """
    Unfreeze the next group of layers in the model.
    Returns the new count of unfrozen layer groups.
    """
    mlp_layers = list(model.mlp)
    # trying to keep frist layer always frozen
    if frozen_layers <= 1:
        return 1

    # Calculate which layer group to unfreeze (each group is 4 layers)
    base_index = (frozen_layers - 1) * 4

    # Unfreeze all four layers in the group
    for i in range(4):
        mlp_layers[base_index + i].requires_grad_(True)

    print(
        f"Unfreezing layer group at index {base_index}: "
        f"{[mlp_layers[base_index + i] for i in range(4)]}"
    )

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
):
    """Fine-tune a pre-trained model on professional game data"""
    device = get_best_device()
    print(f"Using device: {device}")

    # Load patch mapping
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]

    # Initialize unfreezing state
    # We start with 1 unfrozen layer
    frozen_layers = finetune_config.initial_frozen_layers
    last_unfreeze_epoch = 0

    main_model_config = TrainingConfig()
    # Initialize the model
    model = Model(
        config=main_model_config,
        hidden_dims=main_model_config.hidden_dims,
        dropout=finetune_config.dropout,
    )

    model = load_model_state_dict(model, device, path=pretrained_model_path)

    # Initially freeze ALL embeddings including queue_type
    print("Initially freezing all embedding layers...")
    model.patch_embedding.requires_grad_(False)
    model.champion_patch_embedding.requires_grad_(False)
    for name, embedding in model.embeddings.items():
        embedding.requires_grad_(False)
        # if name != "queue_type":
        # embedding.requires_grad_(False)

    # Freeze early MLP layers
    print(f"Freezing first {finetune_config.initial_frozen_layers} MLP layer groups...")
    mlp_layers = list(model.mlp)
    layers_to_freeze = (
        finetune_config.initial_frozen_layers * 4
    )  # Each group has 4 layers
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
    )

    # Check if we're using only pro data
    using_only_pro_data = finetune_config.original_batch_size == 0

    for epoch in range(finetune_config.num_epochs):
        epoch_start = time.time()

        if epoch >= finetune_config.num_epochs * 0.75:
            print(
                f"Enabling last epoch tasks at epoch {epoch} (75% of {finetune_config.num_epochs})"
            )
            enabled_tasks = FINE_TUNE_TASKS_LAST_EPOCHS
            task_names = list(enabled_tasks.keys())
            task_weights = torch.tensor(
                [task_def.weight for task_def in enabled_tasks.values()], device=device
            )

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

        # Progressive unfreezing check
        if (
            finetune_config.progressive_unfreezing
            and frozen_layers > 0
            and epoch - last_unfreeze_epoch >= finetune_config.epochs_per_unfreeze
        ):
            if finetune_config.epochs_per_unfreeze == 200:
                print("Lowering unfreezing frequency to 100 epochs")
                finetune_config.epochs_per_unfreeze = 100
            frozen_layers = unfreeze_layer_group(model, frozen_layers)
            last_unfreeze_epoch = epoch

            # Reinitialize optimizer with newly unfrozen parameters
            optimizer = optim.AdamW(
                get_optimizer_grouped_parameters(model, finetune_config.weight_decay),
                lr=finetune_config.learning_rate,
            )
            print(f"\nUnfroze layer group. Remaining frozen groups: {frozen_layers}")

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
        validate(
            model, val_pro_loader, val_original_loader, finetune_config, device, epoch
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

    # Save the final best model
    # TODO: have logic that saves the best model based on validation loss
    torch.save(model.state_dict(), output_model_path)
    print(f"Final model saved to {output_model_path}")

    if finetune_config.log_wandb:
        wandb.finish()

    return


def validate(
    model: Model,
    val_pro_loader: DataLoader,
    val_original_loader: Optional[DataLoader],
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[Optional[float], Dict[str, float]]:
    """Run validation on both pro and original data"""
    model.eval()
    enabled_tasks = FINE_TUNE_TASKS

    # Initialize accumulators for both datasets
    pro_accumulators = {
        task_name: torch.zeros(2, device=device) for task_name in enabled_tasks.keys()
    }

    # Add accuracy accumulator for win_prediction
    pro_win_accuracy = torch.zeros(2, device=device)

    # Only initialize original accumulators if we have original data
    if val_original_loader is not None:
        original_accumulators = {
            task_name: torch.zeros(2, device=device)
            for task_name in enabled_tasks.keys()
        }
        original_win_accuracy = torch.zeros(2, device=device)

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
        wandb.log(pro_metrics)

        # don't validate on every epoch, it's slow
        if epoch % 50 == 0 and val_original_loader is not None:
            original_metrics = validate_loader(
                model=model,
                loader=val_original_loader,
                accumulators=original_accumulators,
                win_accuracy=original_win_accuracy,
                device=device,
                prefix="val_original",
            )
            wandb.log(original_metrics)

    return


def validate_loader(
    model: Model,
    loader: DataLoader,
    accumulators: Dict[str, torch.Tensor],
    win_accuracy: torch.Tensor,
    device: torch.device,
    prefix: str,
) -> Dict[str, float]:
    """Validate model on a single loader"""
    total_loss = torch.tensor(0.0, device=device)
    total_steps = 0

    task_weights = torch.tensor(
        [task_def.weight for task_def in FINE_TUNE_TASKS.values()], device=device
    )

    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}
        labels = {k: v.to(device) for k, v in labels.items()}
        batch_size = next(iter(labels.values())).size(0)

        outputs = model(features)

        # Calculate task losses
        batch_losses = []
        for task_name, task_def in FINE_TUNE_TASKS.items():
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

            batch_losses.append(loss.mean())

        # Calculate win prediction accuracy
        if "win_prediction" in FINE_TUNE_TASKS:
            win_probs = torch.sigmoid(outputs["win_prediction"])
            predictions = (win_probs > 0.5).float()
            correct = (predictions == labels["win_prediction"]).sum()
            win_accuracy[0] += correct
            win_accuracy[1] += batch_size

        # Calculate total weighted loss
        losses = torch.stack(batch_losses)
        total_loss += (losses * task_weights).sum()
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

    # Add average loss
    metrics[f"{prefix}_avg_loss"] = (total_loss / total_steps).item()

    return metrics


def main():
    """Main function to run fine-tuning"""
    parser = argparse.ArgumentParser(
        description="Fine-tune model on professional game data"
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
    )


if __name__ == "__main__":
    main()
