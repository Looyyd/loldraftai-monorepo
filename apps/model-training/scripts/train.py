# scripts/train.py
import os
import pickle
import glob
import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import argparse


from utils.match_dataset import MatchDataset
from utils.model import MatchOutcomeModel
from utils import (
    get_best_device,
    TRAIN_DIR,
    TEST_DIR,
    ENCODERS_PATH,
    MODEL_PATH,
    TASK_STATS_PATH,
    TRAIN_BATCH_SIZE,
    NUMERICAL_STATS_PATH,
)
from utils.column_definitions import (
    COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    ColumnType,
)
from utils.task_definitions import TASKS, TaskType

DATALOADER_WORKERS = 1 # Fastest with 1 or 2 might be because of mps performance cores

MASK_CHAMPIONS = 0.1

# should use TensorFloat-32, which is faster that "highest" precision
torch.set_float32_matmul_precision("high")

LOG_WANDB = False


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def collate_fn(batch):
    with open(NUMERICAL_STATS_PATH, "rb") as f:
        stats = pickle.load(f)

    means = stats["means"]
    stds = stats["stds"]

    collated = {}
    for col, col_def in COLUMNS.items():
        if col_def.column_type == ColumnType.LIST:
            # For list columns, stack the tensors
            collated[col] = torch.stack(
                [item[col] for item in batch]
            )  # Shape: [batch_size, seq_length]
        elif col_def.column_type == ColumnType.CATEGORICAL:
            # Convert list of integers to a tensor
            collated[col] = torch.tensor(
                [item[col] for item in batch], dtype=torch.long
            )  # Shape: [batch_size]
        elif col_def.column_type == ColumnType.NUMERICAL:
            # Normalize numerical features
            values = torch.tensor(
                [item[col] for item in batch], dtype=torch.float
            )  # Shape: [batch_size]
            mean = means[col]
            std = stds[col] if stds[col] > 0 else 1.0  # Avoid division by zero
            values = (values - mean) / std
            collated[col] = values

    # Collect labels for all tasks
    collated_labels = {}
    for task_name in TASKS.keys():
        task_labels = [item[task_name] for item in batch]
        if TASKS[task_name].task_type == TaskType.BINARY_CLASSIFICATION:
            labels = torch.tensor(task_labels, dtype=torch.float)
        elif TASKS[task_name].task_type == TaskType.REGRESSION:
            labels = torch.tensor(task_labels, dtype=torch.float)
        elif TASKS[task_name].task_type == TaskType.MULTICLASS_CLASSIFICATION:
            labels = torch.tensor(task_labels, dtype=torch.long)
        collated_labels[task_name] = labels

    return collated, collated_labels


def get_max_champion_id():
    max_id = 0
    for dir_path in [TRAIN_DIR, TEST_DIR]:
        data_files = glob.glob(os.path.join(dir_path, "*.parquet"))
        for file_path in data_files:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=10000):
                df_chunk = batch.to_pandas()
                max_id_in_chunk = df_chunk["champion_ids"].apply(max).max()
                if max_id_in_chunk > max_id:
                    max_id = max_id_in_chunk
    return max_id


def train_model(run_name: str):
    # Initialize wandb
    if LOG_WANDB:
        wandb.init(project="draftking", name=run_name)

    # Determine the maximum champion ID
    max_champion_id = get_max_champion_id()
    unknown_champion_id = max_champion_id + 1
    num_champions = unknown_champion_id + 1  # Total number of embeddings

    # Load task statistics
    with open(TASK_STATS_PATH, "rb") as f:
        task_stats = pickle.load(f)
    # Initialize the datasets with masking parameters
    train_dataset = MatchDataset(
        data_dir=TRAIN_DIR,
        mask_champions=MASK_CHAMPIONS,
        unknown_champion_id=unknown_champion_id,
        task_stats=task_stats,
    )
    test_dataset = MatchDataset(
        data_dir=TEST_DIR,
        mask_champions=MASK_CHAMPIONS,
        unknown_champion_id=unknown_champion_id,
        task_stats=task_stats,
    )

    # Initialize the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
    )

    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }

    # Initialize the model
    model = MatchOutcomeModel(
        num_categories=num_categories,
        num_champions=num_champions,
        num_numerical_features=len(NUMERICAL_COLUMNS),
        embed_dim=32,
        dropout=0.1,
    )

    device = get_best_device()
    model.to(device)
    print(f"Using device: {device}")
    if device != torch.device("mps"):
        model = torch.compile(model)

    if LOG_WANDB:
        wandb.watch(model, log_freq=100)

    # Initialize loss functions for each task
    criterion = {}
    for task_name, task_def in TASKS.items():
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            criterion[task_name] = nn.BCELoss()
        elif task_def.task_type == TaskType.REGRESSION:
            criterion[task_name] = nn.MSELoss()
        elif task_def.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            criterion[task_name] = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Add gradient clipping max norm
    max_grad_norm = 1.0

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for batch_idx, (features, labels) in enumerate(train_loader):

            # Move all features to the device
            features = {k: v.to(device) for k, v in features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            # only works with CUDA
            # with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            outputs = model(features)

            total_loss = 0.0
            loss_dict = {}
            for task_name, task_def in TASKS.items():
                task_output = outputs[task_name]
                task_label = labels[task_name]
                task_loss = criterion[task_name](task_output, task_label)
                weighted_loss = task_def.weight * task_loss
                total_loss += weighted_loss
                loss_dict[task_name] = task_loss.item()

            total_loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_steps += 1

            # Logging
            if (batch_idx + 1) % 20 == 0 and LOG_WANDB:
                log_data = {
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "grad_norm": grad_norm,
                }
                log_data.update({f"train_loss_{k}": v for k, v in loss_dict.items()})
                wandb.log(log_data)

            avg_loss = epoch_loss / epoch_steps
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
            if LOG_WANDB:
                wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

        # Evaluation
        model.eval()
        metrics = {task_name: [] for task_name in TASKS.keys()}
        with torch.no_grad():
            for features, labels in test_loader:
                # Move all features to the device
                features = {k: v.to(device) for k, v in features.items()}
                labels = {k: v.to(device) for k, v in labels.items()}

                # TODO: avg_validation_loss
                outputs = model(features)
                for task_name, task_def in TASKS.items():
                    task_output = outputs[task_name]
                    task_label = labels[task_name]

                    if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                        preds = (task_output >= 0.5).float()
                        accuracy = (preds == task_label).float().mean().item()
                        metrics[task_name].append(accuracy)
                    elif task_def.task_type == TaskType.REGRESSION:
                        mse = nn.functional.mse_loss(task_output, task_label).item()
                        metrics[task_name].append(mse)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch time: {epoch_time}")
        if LOG_WANDB:
            wandb.log({"epoch_time": epoch_time})

        # Log evaluation metrics
        for task_name, values in metrics.items():
            avg_metric = sum(values) / len(values)
            if LOG_WANDB:
                wandb.log({f"val_{task_name}_metric": avg_metric})

    if LOG_WANDB:
        wandb.finish()
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train the MatchOutcomeTransformer model"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default="initial-setup",
        help="Name for the Wandb run",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    set_random_seeds()
    train_model(args.run_name)
