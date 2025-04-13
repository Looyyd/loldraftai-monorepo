# scripts/match-prediction/validation.py
# TODO: needs to be updated for new model architecture
import argparse
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from typing import Dict, List, Callable, Any, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.match_prediction import (
    get_best_device,
    MODEL_PATH,
    TRAIN_BATCH_SIZE,
    CHAMPION_ID_ENCODER_PATH,
    PATCH_MAPPING_PATH,
    load_model_state_dict,
)
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.match_dataset import MatchDataset
from utils.match_prediction.model import Model
from utils.match_prediction.train_utils import collate_fn
from utils.match_prediction.task_definitions import TaskDefinition, TaskType
from utils import DATA_DIR, champion_play_rates_path


class FixedMaskCount:
    """Returns a fixed number of champions to mask."""

    def __init__(self, count: int):
        self.count = count

    def __call__(self) -> int:
        return self.count


def get_elo_subgroup(feature: torch.Tensor) -> str:
    """
    Determine the ELO subgroup for a sample based on its ELO feature.

    Parameters:
        feature (torch.Tensor): ELO feature for the sample (scalar tensor).

    Returns:
        str: Subgroup name.
    """
    # feature: scalar tensor
    elo = feature.item()
    subgroup_name = f"elo_{elo}"
    return subgroup_name


with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
    champion_id_encoder: LabelEncoder = pickle.load(f)["mapping"]

with open(PATCH_MAPPING_PATH, "rb") as f:
    patch_mapping = pickle.load(f)["mapping"]

reverse_patch_mapping = {v: k for k, v in patch_mapping.items()}


def get_patch_subgroup(feature: torch.Tensor) -> str:
    """
    Assign a subgroup based on the patch number.
    """
    patch_number = feature.item()
    subgroup_name = f"patch_{patch_number}"
    return subgroup_name


def get_playrate_subgroup_batch(
    champion_ids_batch: torch.Tensor,  # Shape: [batch_size, 10]
    play_rates: Dict[str, Dict[str, Dict[str, float]]],
    patches_batch: torch.Tensor,  # Shape: [batch_size]
) -> List[str]:
    """
    Vectorized version that processes entire batch at once.
    """
    batch_size = champion_ids_batch.shape[0]

    # global play rate buckets
    play_rate_buckets = [
        (0.01, "almost_never"),
        (0.05, "ultra_rare"),
        (0.1, "very_rare"),
        (0.5, "rare"),
        (1.0, "quite_rare"),
        (5.0, "moderately_common"),
        (10.0, "common"),
        (100.0, "very_common"),
    ]
    role_names = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

    # Convert champion IDs for entire batch at once
    champion_ids_flat = champion_ids_batch.view(-1)  # Shape: [batch_size * 10]
    champion_original_ids = champion_id_encoder.inverse_transform(
        champion_ids_flat.cpu().numpy()
    )

    # Initialize play rate tensor
    play_rates_tensor = torch.ones(
        batch_size, 10, dtype=torch.float32
    )  # Shape: [batch_size, 10]

    patches = patches_batch.cpu().numpy()
    for b in range(batch_size):
        patch = reverse_patch_mapping[patches[b]]
        for pos in range(10):
            champ_name = champion_original_ids[b * 10 + pos]
            role = role_names[pos % 5]
            play_rates_tensor[b, pos] = (
                play_rates.get(patch, {}).get(champ_name, {}).get(role, 0.0)
            )

    # Ensure minimum play rate and find minimum per sample
    play_rates_tensor = torch.maximum(play_rates_tensor, torch.tensor(1e-7))
    min_play_rates, _ = play_rates_tensor.min(dim=1)  # Shape: [batch_size]

    # Vectorized bucket assignment
    result = []
    for min_rate in min_play_rates.cpu().numpy():
        for threshold, bucket_name in play_rate_buckets:
            if min_rate <= threshold:
                result.append(f"playrate_{bucket_name}")
                break
        else:
            result.append("playrate_unknown")

    return result


def validate_with_subgroups(
    model: Model,
    test_loader: DataLoader,
    device: torch.device,
    play_rates: Dict[str, Dict[str, Dict[str, float]]],
    test_patch_robustness: bool = False,
) -> pd.DataFrame:
    """
    Validate the model and perform subgroup and calibration analysis.

    Parameters:
        model: The model to validate
        test_loader: DataLoader for test data
        device: The device to run validation on
        play_rates: Dictionary of champion play rates
        test_patch_robustness: If True, simulate using patch N-1 to predict patch N outcomes
    """
    model.eval()

    # For patch robustness testing
    if test_patch_robustness:
        # Find the first/earliest patch index to filter out
        patch_indices = list(patch_mapping.values())
        earliest_patch_index = min(patch_indices)

        # Create mapping of patches to their previous patch
        patch_to_previous = {}
        sorted_patches = sorted(patch_mapping.keys())
        for i in range(1, len(sorted_patches)):
            current_patch = sorted_patches[i]
            previous_patch = sorted_patches[i - 1]
            patch_to_previous[patch_mapping[current_patch]] = patch_mapping[
                previous_patch
            ]

    subgroup_total_loss: Dict[str, float] = defaultdict(float)
    subgroup_counts: Dict[str, int] = defaultdict(int)
    total_samples = 0

    # Add calibration tracking
    confidence_buckets = np.arange(0.5, 1.01, 0.05)  # 0.5, 0.55, ..., 1.0
    calibration_correct = defaultdict(int)  # Tracks correct predictions per bucket
    calibration_total = defaultdict(int)  # Tracks total predictions per bucket

    TASKS = {
        "win_prediction": TaskDefinition(
            name="win_prediction",
            task_type=TaskType.BINARY_CLASSIFICATION,
            weight=1,
        ),
    }

    task_names = list(TASKS.keys())
    task_weights = torch.tensor(
        [TASKS[name].weight for name in task_names], device=device
    )  # Shape: [num_tasks]

    criterions = {}
    # could refactor code in common with train.py
    for task_name, task_def in TASKS.items():
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            criterions[task_name] = nn.BCEWithLogitsLoss(reduction="none")
        elif task_def.task_type == TaskType.REGRESSION:
            criterions[task_name] = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for features_batch, labels_batch in tqdm(
            test_loader,
            desc=f"Validating {'patch robustness' if test_patch_robustness else 'model'}",
        ):
            # Move features and labels to device
            features_batch = {k: v.to(device) for k, v in features_batch.items()}
            labels_batch = {k: labels_batch[k].to(device) for k in TASKS.keys()}

            # For patch robustness testing
            if test_patch_robustness:
                # Filter out samples from the earliest patch
                patch_mask = features_batch["patch"] != earliest_patch_index
                if not patch_mask.any():
                    continue  # Skip this batch if all samples are from the earliest patch

                # Filter the batch to include only valid patches
                for k in features_batch:
                    if (
                        isinstance(features_batch[k], torch.Tensor)
                        and features_batch[k].shape[0] == patch_mask.shape[0]
                    ):
                        features_batch[k] = features_batch[k][patch_mask]

                for k in labels_batch:
                    if (
                        isinstance(labels_batch[k], torch.Tensor)
                        and labels_batch[k].shape[0] == patch_mask.shape[0]
                    ):
                        labels_batch[k] = labels_batch[k][patch_mask]

                if features_batch["patch"].shape[0] == 0:
                    continue  # Skip if no samples left after filtering

                # Store original patches for analysis and future reference
                original_patches = features_batch["patch"].clone()

                # Modify the patch index to simulate using previous patch's data
                modified_patches = torch.zeros_like(features_batch["patch"])
                for i, patch in enumerate(features_batch["patch"]):
                    patch_idx = patch.item()
                    modified_patches[i] = patch_to_previous.get(patch_idx, patch_idx)

                # Replace the patch feature with the modified version
                features_batch["patch"] = modified_patches

            batch_size = labels_batch["win_prediction"].size(0)
            total_samples += batch_size

            # Compute model outputs
            outputs_batch = model(features_batch)

            # Process calibration statistics for win prediction
            win_preds = torch.sigmoid(
                outputs_batch["win_prediction"].squeeze()
            )  # [batch_size]
            win_labels = labels_batch["win_prediction"].squeeze()  # [batch_size]

            # Determine correctness of predictions
            predicted_classes = (win_preds >= 0.5).float()
            is_correct = predicted_classes == win_labels

            # Get the model's confidence in its prediction (always >= 0.5)
            confidences = torch.where(win_preds >= 0.5, win_preds, 1 - win_preds)

            # Process predictions by confidence
            for i in range(len(confidence_buckets) - 1):
                lower, upper = confidence_buckets[i], confidence_buckets[i + 1]
                bucket_name = f"{lower:.2f}-{upper:.2f}"

                # Find predictions that fall into this bucket
                mask = (confidences >= lower) & (confidences < upper)
                calibration_total[bucket_name] += mask.sum().item()
                calibration_correct[bucket_name] += (mask & is_correct).sum().item()

            # Compute losses for entire batch at once
            batch_losses = []
            for task_name in task_names:
                output = outputs_batch[task_name].view(
                    batch_size, -1
                )  # [batch_size, feat_dim]
                label = labels_batch[task_name].view(
                    batch_size, -1
                )  # [batch_size, feat_dim]
                task_loss = criterions[task_name](output, label)  # [batch_size]
                batch_losses.append(task_loss)

            # Stack and compute total loss for all samples at once
            batch_losses_tensor = torch.stack(
                batch_losses, dim=1
            )  # [batch_size, num_tasks]
            total_losses = (batch_losses_tensor * task_weights).sum(
                dim=1
            )  # [batch_size]

            # Vectorized subgroup assignment
            elo_subgroups = [f"elo_{elo}" for elo in features_batch["elo"].tolist()]

            # Use original patches for subgroup assignment if in robustness testing mode,
            # otherwise use current patches (which may be modified)
            patch_feature = (
                original_patches if test_patch_robustness else features_batch["patch"]
            )
            patch_subgroups = [f"patch_{patch}" for patch in patch_feature.tolist()]

            playrate_subgroups = get_playrate_subgroup_batch(
                features_batch["champion_ids"], play_rates, patch_feature
            )

            # Process all subgroups at once using numpy operations
            all_subgroups = elo_subgroups + patch_subgroups + playrate_subgroups
            all_losses = total_losses.repeat_interleave(
                3
            )  # Repeat each loss 3 times for the 3 subgroup types

            for sg, loss in zip(all_subgroups, all_losses):
                subgroup_total_loss[sg] += loss.item()
                subgroup_counts[sg] += 1

    # Prepare results
    subgroup_data = {
        "mean_loss": {
            k: subgroup_total_loss[k] / subgroup_counts[k] for k in subgroup_total_loss
        },
        "count": subgroup_counts,
        "percentage": {
            k: 100.0 * subgroup_counts[k] / total_samples for k in subgroup_counts
        },
    }

    # Add calibration results
    calibration_data = {
        "confidence_bucket": [],
        "samples": [],
        "accuracy": [],
        "expected_accuracy": [],
    }

    for i in range(len(confidence_buckets) - 1):
        lower, upper = confidence_buckets[i], confidence_buckets[i + 1]
        bucket_name = f"{lower:.2f}-{upper:.2f}"

        if calibration_total[bucket_name] > 0:
            calibration_data["confidence_bucket"].append(
                f"{lower*100:.0f}-{upper*100:.0f}%"
            )
            calibration_data["samples"].append(calibration_total[bucket_name])
            calibration_data["accuracy"].append(
                calibration_correct[bucket_name] / calibration_total[bucket_name]
            )
            calibration_data["expected_accuracy"].append((lower + upper) / 2)

    # Combine all results into a single DataFrame
    df = pd.DataFrame(subgroup_data)
    df_calibration = pd.DataFrame(calibration_data)

    # Add a type column to distinguish between subgroup and calibration results
    df["analysis_type"] = "subgroup"
    df_calibration["analysis_type"] = "calibration"

    # Combine the DataFrames
    combined_df = pd.concat([df, df_calibration], axis=0)

    return combined_df


def main():
    """
    Main function to perform validation and subgroup analysis.
    """
    parser = argparse.ArgumentParser(description="Validate the match-prediction model")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use only a small subset of the dataset for quick iteration",
    )
    parser.add_argument(
        "--test-patch-robustness",
        action="store_true",
        help="Test model robustness by using patch N-1 to predict patch N outcomes",
    )
    parser.add_argument(
        "--mask-champions",
        type=int,
        default=0,
        help="Number of champions to randomly mask in each team composition (default: 0)",
    )
    args = parser.parse_args()

    device = get_best_device()

    # Initialize model
    config = TrainingConfig()
    model = Model(config=config, dropout=0, hidden_dims=config.hidden_dims)
    model = load_model_state_dict(model, device=device, path=MODEL_PATH)
    model.eval()

    # Get the unknown champion ID from the encoder
    unknown_champion_id = champion_id_encoder.transform(["UNKNOWN"])[0]

    # Define masking function if needed
    masking_function = None
    if args.mask_champions > 0:
        masking_function = FixedMaskCount(args.mask_champions)
        print(f"Masking {args.mask_champions} champions in each team composition")

    # Initialize test dataset and loader
    test_dataset = MatchDataset(
        train_or_test="test",
        dataset_fraction=0.01 if args.small else 1.0,
        masking_function=masking_function,
        unknown_champion_id=unknown_champion_id,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    with open(champion_play_rates_path, "r") as f:
        play_rates = json.load(f)

    # Run validation and subgroup analysis
    results_df = validate_with_subgroups(
        model,
        test_loader,
        device,
        play_rates,
        test_patch_robustness=args.test_patch_robustness,
    )

    # Determine output filename based on mode
    base_filename = "validation_results"
    if args.test_patch_robustness:
        base_filename = "validation_patch_robustness_results"
    if args.mask_champions > 0:
        base_filename = f"{base_filename}_masked_{args.mask_champions}"

    output_file = DATA_DIR + "/" + base_filename + ".csv"

    # Print and save results
    print("\nValidation Results:")
    print(results_df)

    # Save results to CSV
    results_df.to_csv(output_file)
    print(f"\nSaved validation results to '{output_file}'")


if __name__ == "__main__":
    main()
