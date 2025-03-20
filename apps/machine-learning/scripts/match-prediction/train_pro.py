#!/usr/bin/env python3

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

from utils.match_prediction import (
    get_best_device,
    load_model_state_dict,
    RAW_PRO_GAMES_FILE,
    PATCH_MAPPING_PATH,
    CHAMPION_ID_ENCODER_PATH,
)
from utils.match_prediction.model import Model
from utils.match_prediction.train_utils import (
    set_random_seeds,
)
from utils.match_prediction.config import (
    TrainingConfig,
)
from utils.match_prediction.column_definitions import (
    RANKED_QUEUE_INDEX,
    PRO_QUEUE_INDEX,
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
        # Fine-tuning hyperparameters - edit these directly instead of using command line flags
        self.num_epochs = 60
        self.learning_rate = 5e-6  # Lower learning rate for fine-tuning
        self.weight_decay = 0.05  # Much stronger regularization
        self.dropout = 0.3  # Higher dropout to prevent overfitting
        self.batch_size = 16
        self.val_split = 0.2
        self.max_grad_norm = 1.0
        self.log_wandb = True
        self.log_batch_interval = 5  # How often to log batch progress
        self.save_checkpoints = False

        # New unfreezing parameters
        self.progressive_unfreezing = True  # Enable progressive unfreezing
        self.epochs_per_unfreeze = 10  # Number of epochs before unfreezing next layer
        self.initial_frozen_layers = (
            6  # Number of layers to start frozen (2 linear + 2 batchnorm)
        )

        # Data augmentation options
        self.use_team_symmetry = True  # Enable team symmetry augmentation
        self.team_symmetry_epochs = (
            self.num_epochs
        )  # Number of epochs to use team symmetry

        # Label smoothing options
        self.use_label_smoothing = True  # Enable label smoothing
        self.smooth_low = 0.1  # Value to smooth 0 labels to
        self.smooth_high = 0.9  # Value to smooth 1 labels to

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

        self.use_team_symmetry = use_team_symmetry and train_or_test == "train"

        # Label smoothing parameters
        self.use_label_smoothing = use_label_smoothing and train_or_test == "train"
        self.smooth_low = smooth_low
        self.smooth_high = smooth_high

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

        # Tensor 2 is reserved for pro play, see column_definitions.py
        features["queue_type"] = torch.tensor(PRO_QUEUE_INDEX, dtype=torch.long)
        # Add numerical_elo = 0 for pro games (highest skill level)
        features["elo"] = torch.tensor(0.0, dtype=torch.long)

        # Extract labels with symmetry handling
        win_prediction = row["team_100_win"]
        if use_symmetry:
            win_prediction = not win_prediction

        # Apply label smoothing only to binary labels
        if self.use_label_smoothing and win_prediction in [0, 1]:
            win_prediction = (
                self.smooth_low if win_prediction == 0 else self.smooth_high
            )

        labels = {"win_prediction": torch.tensor(win_prediction, dtype=torch.float32)}

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


def create_dataloaders(
    pro_games_df,
    patch_mapping,
    config,
    current_epoch: int = 0,  # Add epoch parameter to control team symmetry
):
    """Create train and validation dataloaders for fine-tuning"""
    # Determine if we should use team symmetry based on the current epoch
    use_symmetry = (
        config.use_team_symmetry and current_epoch < config.team_symmetry_epochs
    )

    # Create datasets
    train_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        patch_mapping=patch_mapping,
        train_or_test="train",
        val_split=config.val_split,
        use_team_symmetry=use_symmetry,  # Use symmetry based on current epoch
        use_label_smoothing=config.use_label_smoothing,
        smooth_low=config.smooth_low,
        smooth_high=config.smooth_high,
    )

    val_dataset = ProMatchDataset(
        pro_games_df=pro_games_df,
        patch_mapping=patch_mapping,
        train_or_test="test",
        val_split=config.val_split,
        use_team_symmetry=False,
        use_label_smoothing=False,
        smooth_low=config.smooth_low,
        smooth_high=config.smooth_high,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=pro_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=pro_collate_fn,
    )

    return train_loader, val_loader


def unfreeze_layer_group(
    model: Model, current_unfrozen_layers: int, config: FineTuningConfig
) -> int:
    """
    Unfreeze the next group of layers in the model.
    Returns the new count of unfrozen layer groups.
    """
    mlp_layers = list(model.mlp)
    total_layer_groups = len(mlp_layers) // 2  # Each group has linear + batchnorm

    if current_unfrozen_layers >= total_layer_groups:
        return current_unfrozen_layers

    # Calculate which layer group to unfreeze
    layer_to_unfreeze = -(current_unfrozen_layers + 1) * 2

    # Unfreeze the next linear + batchnorm group
    mlp_layers[layer_to_unfreeze].requires_grad_(True)
    mlp_layers[layer_to_unfreeze + 1].requires_grad_(True)

    print(
        f"Unfreezing layer group {total_layer_groups - current_unfrozen_layers}: "
        f"{type(mlp_layers[layer_to_unfreeze]).__name__} and "
        f"{type(mlp_layers[layer_to_unfreeze + 1]).__name__}"
    )

    return current_unfrozen_layers + 1


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
    current_unfrozen_layers = 0
    last_unfreeze_epoch = 0

    main_model_config = TrainingConfig()
    # Initialize the model
    model = Model(
        embed_dim=main_model_config.embed_dim,
        hidden_dims=main_model_config.hidden_dims,
        dropout=finetune_config.dropout,
    )

    model = load_model_state_dict(model, device, path=pretrained_model_path)

    # Initialize pro play embedding (index 2) with ranked solo/duo embedding (index 0)
    print("Initializing pro play embedding with ranked solo/duo embedding values...")
    with torch.no_grad():
        ranked_solo_embedding = (
            model.embeddings["queue_type"].weight[RANKED_QUEUE_INDEX].clone()
        )
        model.embeddings["queue_type"].weight[PRO_QUEUE_INDEX] = ranked_solo_embedding

    # Freeze embeddings except queueId
    print("Freezing embedding layers except queue_type...")
    model.patch_embedding.requires_grad_(False)
    model.champion_patch_embedding.requires_grad_(False)
    for name, embedding in model.embeddings.items():
        if name != "queue_type":
            embedding.requires_grad_(False)

    # Freeze early MLP layers
    print(f"Freezing first {finetune_config.initial_frozen_layers} MLP layers...")
    mlp_layers = list(model.mlp)
    for layer in mlp_layers[: finetune_config.initial_frozen_layers]:
        print(f"Freezing layer: {layer}")
        layer.requires_grad_(False)

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

    # Define loss function for win prediction only
    criterion = {"win_prediction": nn.BCEWithLogitsLoss()}

    # Initialize optimizer with initially trainable parameters
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=finetune_config.learning_rate,
        weight_decay=finetune_config.weight_decay,
    )

    for epoch in range(finetune_config.num_epochs):
        # Create dataloaders for this epoch - this allows team symmetry  to change based on epoch
        train_loader, val_loader = create_dataloaders(
            pro_games_df,
            patch_mapping,
            finetune_config,
            current_epoch=epoch,  # Pass current epoch to control team symmetry
        )

        # Print team symmetry status at the start of each epoch
        is_using_symmetry = epoch < finetune_config.team_symmetry_epochs

        print(
            f"Epoch {epoch+1}/{finetune_config.num_epochs} - "
            f"Team symmetry: {'enabled' if is_using_symmetry else 'disabled'}, "
        )

        # If this is the epoch where team log it more prominently
        if epoch == finetune_config.team_symmetry_epochs:
            print(f"\n=== DISABLING TEAM SYMMETRY AUGMENTATION AT EPOCH {epoch} ===\n")

            # Also log to wandb if enabled
            if finetune_config.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "team_symmetry_disabled": True,
                    }
                )

        # Check if it's time to unfreeze next layer group
        if (
            finetune_config.progressive_unfreezing
            and epoch > 0
            and epoch - last_unfreeze_epoch >= finetune_config.epochs_per_unfreeze
        ):

            current_unfrozen_layers = unfreeze_layer_group(
                model, current_unfrozen_layers, finetune_config
            )
            last_unfreeze_epoch = epoch

            # Reconfigure optimizer with updated trainable parameters
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=finetune_config.learning_rate,
                weight_decay=finetune_config.weight_decay,
            )

            # Log unfreezing event
            if finetune_config.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "unfrozen_layer_groups": current_unfrozen_layers,
                        "trainable_params": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                    }
                )

        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion["win_prediction"](
                outputs["win_prediction"], labels["win_prediction"]
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # Calculate accuracy
            probs = torch.sigmoid(outputs["win_prediction"])
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels["win_prediction"]).sum().item()
            train_total += labels["win_prediction"].size(0)

            # Log batch progress - minimal console output
            if batch_idx % finetune_config.log_batch_interval == 0:
                print(
                    f"Epoch {epoch+1}/{finetune_config.num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / train_steps
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            # Regular validation
            for features, labels in val_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(features)
                loss = criterion["win_prediction"](
                    outputs["win_prediction"], labels["win_prediction"]
                )

                val_loss += loss.item()
                val_steps += 1

                # Calculate accuracy
                probs = torch.sigmoid(outputs["win_prediction"])
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels["win_prediction"]).sum().item()
                val_total += labels["win_prediction"].size(0)

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float("inf")
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        epoch_time = time.time() - epoch_start

        # Enhanced console output to show both loss and accuracy
        print(
            f"Epoch {epoch+1}/{finetune_config.num_epochs} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}\n"
            f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n"
        )

        # Log metrics to wandb
        if finetune_config.log_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch_time": epoch_time,
                    "team_symmetry_active": is_using_symmetry,
                }
            )

    # Save the final best model
    torch.save(model.state_dict(), output_model_path)
    print(f"Final model saved to {output_model_path}")

    if finetune_config.log_wandb:
        wandb.finish()

    return avg_val_loss  # Return best validation loss instead of accuracy


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
    best_val_loss = fine_tune_model(
        pretrained_model_path=args.model_path,
        pro_games_df=pro_games_df,
        finetune_config=config,
        output_model_path=args.output_path,
        run_name=args.run_name,
    )

    print(
        f"Fine-tuning complete. Best model saved to {args.output_path} with validation loss {best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
