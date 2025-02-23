import argparse
import pickle
from collections import defaultdict
from typing import Dict, List, Callable, Any, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.match_prediction import (
    get_best_device,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_BATCH_SIZE,
    ENCODERS_PATH,
)
from utils.match_prediction.column_definitions import CATEGORICAL_COLUMNS
from utils.match_prediction.match_dataset import MatchDataset
from utils.match_prediction.model import Model
from utils.match_prediction.task_definitions import TASKS
from utils.match_prediction.train import get_num_champions, collate_fn
from utils import DATA_DIR


def calculate_play_rates(data_loader: DataLoader) -> Dict[Tuple[int, int], float]:
    """
    Calculate play rates for each champion-role combination.

    Parameters:
        data_loader (DataLoader): DataLoader for the dataset.

    Returns:
        Dict[Tuple[int, int], float]: Play rates for each (champion_id, role_idx).
    """
    play_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total_counts_per_role: Dict[int, int] = defaultdict(int)
    total_games = 0

    print("Calculating play rates...")
    for features_batch, _ in tqdm(data_loader):
        champion_ids_batch = features_batch["champion_ids"]  # Shape: [batch_size, 10]
        batch_size = champion_ids_batch.size(0)
        total_games += batch_size

        for champion_ids in champion_ids_batch:
            # champion_ids: Tensor of shape [10]
            for pos_idx, champ_id in enumerate(champion_ids):
                role_idx = (
                    pos_idx % 5
                )  # Positions 0-4 are team1 roles, 5-9 are team2 roles
                champ_id = champ_id.item()
                play_counts[(champ_id, role_idx)] += 1
                total_counts_per_role[role_idx] += 1

    # Calculate play rates per role
    play_rates = {
        pair: count / total_counts_per_role[pair[1]]
        for pair, count in play_counts.items()
    }

    print("\nCalculated play rates for champion-role combinations.")
    return play_rates


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
    bucket = round(elo * 10) / 10  # Round to nearest 0.1
    subgroup_name = f"elo_{bucket}"
    return subgroup_name


def get_patch_subgroup(feature: torch.Tensor) -> str:
    """
    Assign a subgroup based on the patch number.
    """
    patch_number = feature.item()
    subgroup_name = f"patch_{patch_number}"
    return subgroup_name


def get_playrate_subgroup(
    champion_ids: torch.Tensor, play_rates: Dict[Tuple[int, int], float]
) -> str:
    """
    Assign a subgroup based on the rarest champion in the game.

    Parameters:
        champion_ids (torch.Tensor): Tensor of champion IDs for the sample. Shape: [10]
        play_rates (Dict[Tuple[int, int], float]): Play rates for champion-role combinations.

    Returns:
        str: Subgroup name.
    """
    # Define play rate buckets (in decimals), ordered from lowest to highest threshold
    play_rate_buckets = [
        (0.000001, "almost_never"),  # <= 0.001%
        (0.00005, "ultra_rare"),  # <= 0.001%
        (0.0001, "very_rare"),  # <= 0.01%
        (0.0005, "rare"),  # <= 0.05%
        (0.001, "quite_rare"),  # <= 0.1%
        (0.005, "moderately_common"),  # <= 0.5%
        (0.01, "common"),  # <= 1%
        (1.0, "very_common"),  # > 1%
    ]

    min_play_rate = 1.0  # Initialize with maximum possible play rate

    # Find the minimal play rate among champions in the game
    for pos_idx, champ_id in enumerate(champion_ids):
        role_idx = pos_idx % 5
        champ_id = champ_id.item()
        play_rate = play_rates.get((champ_id, role_idx), 0.0)
        play_rate = max(play_rate, 1e-7)  # Avoid zero play rates
        if play_rate < min_play_rate:
            min_play_rate = play_rate

    # Determine the bucket for the minimal play rate
    for threshold, bucket_name in play_rate_buckets:
        if min_play_rate <= threshold:
            subgroup_name = f"playrate_{bucket_name}"
            return subgroup_name
    return "playrate_unknown"


def validate_with_subgroups(
    model: Model,
    test_loader: DataLoader,
    device: torch.device,
    play_rates: Dict[Tuple[int, int], float],
) -> pd.DataFrame:
    """
    Validate the model and perform subgroup analysis.

    Parameters:
        model (Model): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the computations on.
        play_rates (Dict[Tuple[int, int], float]): Play rates for champion-role combinations.

    Returns:
        pd.DataFrame: DataFrame containing the subgroup analysis results.
    """
    model.eval()

    subgroup_total_loss: Dict[str, float] = defaultdict(float)
    subgroup_counts: Dict[str, int] = defaultdict(int)
    total_samples = 0

    task_names = list(TASKS.keys())
    task_weights = torch.tensor(
        [TASKS[name].weight for name in task_names], device=device
    )  # Shape: [num_tasks]

    loss_fn = nn.MSELoss(reduction="mean")

    with torch.no_grad():
        for features_batch, labels_batch in tqdm(test_loader, desc="Validating"):
            # Move features and labels to device
            features_batch = {k: v.to(device) for k, v in features_batch.items()}
            labels_batch = {k: v.to(device) for k, v in labels_batch.items()}

            batch_size = next(iter(labels_batch.values())).size(
                0
            )  # Number of samples in batch
            total_samples += batch_size

            # Compute model outputs
            outputs_batch = model(features_batch)  # Outputs are dict of tensors

            # For each sample in the batch
            for i in range(batch_size):
                # Extract sample data
                sample_features = {k: v[i] for k, v in features_batch.items()}
                sample_labels = {k: v[i] for k, v in labels_batch.items()}
                sample_outputs = {k: v[i] for k, v in outputs_batch.items()}

                # Compute loss per task
                sample_losses = []
                for task_name in task_names:
                    output = sample_outputs[task_name]
                    label = sample_labels[task_name]
                    # If the output is multi-dimensional, reduce it
                    if output.dim() > 0:
                        output = output.view(-1)  # Flatten
                        label = label.view(-1)  # Flatten
                    task_loss = loss_fn(output, label)
                    sample_losses.append(task_loss)

                # Total loss for the sample
                sample_losses_tensor = torch.stack(sample_losses)  # Shape: [num_tasks]
                total_loss = (sample_losses_tensor * task_weights).sum().item()

                # Get subgroups for the sample
                elo_subgroup = get_elo_subgroup(
                    sample_features["numerical_elo"]
                )  # Shape: scalar
                playrate_subgroup = get_playrate_subgroup(
                    sample_features["champion_ids"], play_rates
                )  # Shape: scalar
                patch_subgroup = get_patch_subgroup(sample_features["numerical_patch"])

                # Accumulate loss and counts for each subgroup
                for subgroup_name in [elo_subgroup, playrate_subgroup, patch_subgroup]:
                    subgroup_total_loss[subgroup_name] += total_loss
                    subgroup_counts[subgroup_name] += 1

    # Prepare results
    data = {
        "mean_loss": {
            k: subgroup_total_loss[k] / subgroup_counts[k] for k in subgroup_total_loss
        },
        "count": subgroup_counts,
        "percentage": {
            k: 100.0 * subgroup_counts[k] / total_samples for k in subgroup_counts
        },
    }
    df = pd.DataFrame(data)

    return df


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
    args = parser.parse_args()

    device = get_best_device()

    # Load model config
    with open(MODEL_CONFIG_PATH, "rb") as f:
        model_config = pickle.load(f)

    # Load encoders to get number of categories
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    num_champions, _ = get_num_champions()

    # Initialize model
    model = Model(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=model_config["embed_dim"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Initialize test dataset and loader
    test_dataset = MatchDataset(train_or_test="test", small_dataset=args.small)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Calculate play rates
    play_rates = calculate_play_rates(test_loader)

    # Run validation and subgroup analysis
    results_df = validate_with_subgroups(model, test_loader, device, play_rates)

    # Print and save results
    print("\nValidation Results:")
    print(results_df)

    # Save results to CSV
    results_df.to_csv(DATA_DIR + "/validation_results.csv")
    print("\nSaved validation results to 'validation_results.csv'")


if __name__ == "__main__":
    main()
