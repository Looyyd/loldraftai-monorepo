# scripts/train.py
import multiprocessing
import signal
import os
import copy
import pickle
import datetime
import time
import cProfile
import pstats
import io
from typing import Any, Dict, List, Optional, Tuple
from pstats import SortKey

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.data import DataLoader
import argparse

from torch.optim.lr_scheduler import OneCycleLR

from utils.match_prediction.match_dataset import MatchDataset
from utils.match_prediction.model import SimpleMatchModel
from utils import DATA_DIR
from utils.match_prediction import (
    get_best_device,
    ENCODERS_PATH,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_BATCH_SIZE,
)
from utils.match_prediction.column_definitions import (
    CATEGORICAL_COLUMNS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType, get_enabled_tasks
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.train import (
    get_optimizer_grouped_parameters,
    set_random_seeds,
    get_num_champions,
    collate_fn,
)
import psutil

# Enable cuDNN auto-tuner, source https://x.com/karpathy/status/1299921324333170689/photo/1
torch.backends.cudnn.benchmark = True
# should use TensorFloat-32, which is faster that "highest" precision
torch.set_float32_matmul_precision("high")


device = get_best_device()


def get_dataloader_config():
    if device.type == "cuda":
        # Original CUDA config
        num_cpus = multiprocessing.cpu_count()
        return {
            "num_workers": max(1, min(num_cpus - 1, 8)),
            "prefetch_factor": 2,
            "pin_memory": True,
        }
    elif device.type == "mps":
        # M1/M2 Mac config
        return {
            "num_workers": 1,
            "prefetch_factor": 1,
            "pin_memory": True,
        }
    else:  # CPU
        # For 4 CPU machine, reserve 1 CPU for main process
        total_cpus = multiprocessing.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use 2 workers for 4 CPU machine, leaving 2 CPUs for main process and OS
        # Adjust prefetch_factor based on available memory
        return {
            "num_workers": min(total_cpus - 2, 2),  # Use 2 workers on 4 CPU machine
            "prefetch_factor": (
                2 if available_memory_gb > 8 else 1
            ),  # Lower if memory constrained
            "pin_memory": False,  # False for CPU training
        }


# Use in DataLoader initialization
dataloader_config = get_dataloader_config()


def save_model(model: SimpleMatchModel, timestamp: Optional[str] = None) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_timestamp_path = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pth"
    torch.save(model.state_dict(), model_timestamp_path)
    print(f"Model saved to {model_timestamp_path}")
    return model_timestamp_path


def cleanup() -> None:
    if "model" in globals():
        save_model(model)
    if wandb.run is not None:
        wandb.finish()


def signal_handler(signum: int, frame: Any) -> None:
    print("Received interrupt signal. Saving model and exiting...")
    cleanup()
    exit(0)


def init_model(
    config: TrainingConfig,
    num_champions: int,
    continue_training: bool,
    load_path: Optional[str] = None,
):
    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    # Initialize the model
    model = SimpleMatchModel(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=config.embed_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    if continue_training:
        load_path = load_path or MODEL_PATH
        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            model.load_state_dict(torch.load(load_path, weights_only=True))
        else:
            print(f"No saved model found at {load_path}. Starting from scratch.")

    # Create a dictionary with model parameters
    model_params = {
        "num_categories": num_categories,
        "num_champions": num_champions,
        **config.to_dict(),
    }

    # Save model config
    with open(MODEL_CONFIG_PATH, "wb") as f:
        pickle.dump(model_params, f)

    model.to(device)
    if device == torch.device("cuda"):
        print("Compiling model")
        model = torch.compile(model)
        print("Model compiled")

    if config.log_wandb:
        wandb.watch(model, log_freq=1000)

    return model


def apply_label_smoothing(labels: torch.Tensor, smoothing: float = 0.2) -> torch.Tensor:
    # Apply label smoothing
    return labels * (1 - smoothing) + 0.5 * smoothing


def train_epoch(
    model: SimpleMatchModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[OneCycleLR],
    criterion: Dict[str, nn.Module],
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[float, int]:
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    enabled_tasks = get_enabled_tasks(config)
    task_names = list(enabled_tasks.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in enabled_tasks.values()], device=device
    )

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
        labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

        # Apply label smoothing to binary classification labels
        for task_name, task_def in TASKS.items():
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                labels[task_name] = apply_label_smoothing(labels[task_name])

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(features)
            losses = torch.stack(
                [
                    criterion[task_name](outputs[task_name], labels[task_name])
                    for task_name in task_names
                ]
            )
            total_loss = (losses * task_weights).sum() / config.accumulation_steps

        total_loss.backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            # TODO: not sure why but had an error on remote machine where it tried to do an extra step.
            if scheduler is not None and batch_idx < len(train_loader) - 1:
                scheduler.step()
            optimizer.zero_grad()

            if (
                (batch_idx + 1) // config.accumulation_steps
            ) % 20 == 0 and config.log_wandb:
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler
                    else optimizer.param_groups[0]["lr"]
                )
                log_training_step(
                    epoch, batch_idx, grad_norm, losses, task_names, config, current_lr
                )

        epoch_loss += total_loss.item()
        epoch_steps += 1

    return epoch_loss, epoch_steps


def validate(
    model: SimpleMatchModel,
    test_loader: DataLoader,
    criterion: Dict[str, nn.Module],
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[Optional[float], Dict[str, float]]:
    model.eval()
    enabled_tasks = get_enabled_tasks(config)  # Get only enabled tasks
    metric_accumulators = {
        task_name: torch.zeros(2).to(device) for task_name in enabled_tasks.keys()
    }
    num_samples = 0
    total_loss = 0.0
    total_steps = 0
    task_names = list(enabled_tasks.keys())
    task_weights = torch.tensor(
        [task_def.weight for task_def in enabled_tasks.values()], device=device
    )

    with torch.no_grad():
        for features, labels in test_loader:
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features)
                if config.calculate_val_loss:
                    losses = torch.stack(
                        [
                            criterion[task_name](outputs[task_name], labels[task_name])
                            for task_name in task_names
                        ]
                    )
                    total_loss += (losses * task_weights).sum()

            batch_size = next(iter(labels.values())).size(0)
            num_samples += batch_size
            total_steps += 1

            update_metric_accumulators(
                metric_accumulators, outputs, labels, batch_size, config
            )

    metrics = calculate_final_metrics(metric_accumulators)
    avg_loss = total_loss / total_steps if config.calculate_val_loss else None

    return avg_loss, metrics


def log_training_step(
    epoch: int,
    batch_idx: int,
    grad_norm: float,
    losses: torch.Tensor,
    task_names: List[str],
    config: TrainingConfig,
    current_lr: float,
) -> None:
    if config.log_wandb:
        log_data = {
            "epoch": epoch + 1,
            "batch": (batch_idx + 1) // config.accumulation_steps,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
        }
        log_data.update(
            {f"train_loss_{k}": v.item() for k, v in zip(task_names, losses)}
        )
        wandb.log(log_data)


def update_metric_accumulators(
    metric_accumulators: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    batch_size: int,
    config: TrainingConfig,
) -> None:
    for task_name, task_def in TASKS.items():
        if config.calculate_val_win_prediction_only and task_name != "win_prediction":
            continue
        task_output = outputs[task_name]
        task_label = labels[task_name]

        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            probs = torch.sigmoid(task_output)
            preds = (probs >= 0.5).float()
            correct = (preds == task_label).float().sum()
            metric_accumulators[task_name][0] += correct
            metric_accumulators[task_name][1] += batch_size
        elif task_def.task_type == TaskType.REGRESSION:
            mse = nn.functional.mse_loss(task_output, task_label, reduction="sum")
            metric_accumulators[task_name][0] += mse
            metric_accumulators[task_name][1] += batch_size


def calculate_final_metrics(
    metric_accumulators: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    # Calculate final metrics
    metrics = {}
    for task_name, accumulator in metric_accumulators.items():
        if TASKS[task_name].task_type == TaskType.BINARY_CLASSIFICATION:
            metrics[task_name] = (accumulator[0] / accumulator[1]).item()
        elif TASKS[task_name].task_type == TaskType.REGRESSION:
            metrics[task_name] = (accumulator[0] / accumulator[1]).item()
    return metrics


def log_validation_metrics(
    val_metrics: Dict[str, float], config: TrainingConfig
) -> None:
    if config.log_wandb:
        log_data = {}
        for task_name, metric_value in val_metrics.items():
            if (
                config.calculate_val_win_prediction_only
                and task_name != "win_prediction"
            ):
                continue
            log_data[f"val_{task_name}_metric"] = metric_value
        wandb.log(log_data)


def train_model(
    run_name: str | None,
    config: TrainingConfig,
    continue_training: bool = False,
    load_path: Optional[str] = None,
):
    global model  # to be able to save the model on interrupt
    best_metric = float("inf")  # For loss minimization
    best_model_state = None

    num_champions, unknown_champion_id = get_num_champions()

    # Initialize the datasets with masking parameters and dataset fraction
    datasets = []
    for split in ["train", "test", "test_masked"]:
        masking_function = (
            config.get_masking_function() if split in ["train", "test_masked"] else None
        )
        train_or_test = "test" if split.startswith("test") else "train"

        dataset = MatchDataset(
            masking_function=masking_function,
            unknown_champion_id=unknown_champion_id,
            train_or_test=train_or_test,
            dataset_fraction=config.dataset_fraction,
        )
        datasets.append(dataset)

    train_dataset, test_dataset, test_masked_dataset = datasets

    # Initialize wandb
    if config.log_wandb:
        config_dict = config.to_dict()
        config_dict["num_samples"] = len(
            train_dataset
        )  # TODO: this could take some time, maybe we should have a file with this stat and claculate it only once?
        wandb.init(project="draftking", name=run_name, config=config_dict)

    train_loader, test_loader, test_masked_loader = (
        DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            **dataloader_config,
            collate_fn=collate_fn,
        )
        for dataset in [train_dataset, test_dataset, test_masked_dataset]
    )

    model = init_model(config, num_champions, continue_training, load_path)

    # Initialize loss functions for each task
    criterion = {}
    enabled_tasks = get_enabled_tasks(config)  # Get only enabled tasks
    for task_name, task_def in enabled_tasks.items():
        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
            criterion[task_name] = nn.BCEWithLogitsLoss()
        elif task_def.task_type == TaskType.REGRESSION:
            criterion[task_name] = nn.MSELoss()
        elif task_def.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            criterion[task_name] = nn.CrossEntropyLoss()

    fused = True if device.type == "cuda" else False
    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, config.weight_decay),
        lr=config.learning_rate,
        fused=fused,
    )

    # Initialize scheduler
    if config.use_one_cycle_lr:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
            anneal_strategy="cos",  # cosine annealing
        )

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        epoch_loss, epoch_steps = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler if config.use_one_cycle_lr else None,  # Pass scheduler
            criterion,
            config,
            device,
            epoch,
        )
        avg_loss = epoch_loss / epoch_steps * config.accumulation_steps

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {avg_loss:.4f}")
        if config.log_wandb:
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

        # Only run validation on specified intervals
        if (epoch + 1) % config.validation_interval == 0:
            val_loss, val_metrics = validate(
                model, test_loader, criterion, config, device
            )
            val_masked_loss, val_masked_metrics = validate(
                model, test_masked_loader, criterion, config, device
            )

            if config.calculate_val_loss:
                if config.log_wandb:
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_masked_loss": val_masked_loss,
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                            **{
                                f"val_masked_{k}": v
                                for k, v in val_masked_metrics.items()
                            },
                        }
                    )

                if val_loss < best_metric:
                    best_metric = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    torch.save(best_model_state, MODEL_PATH)
                    print(
                        f"New best model saved with validation loss: {best_metric:.4f}"
                    )

            log_validation_metrics(val_metrics, config)

        epoch_end_time = time.time()
        if config.log_wandb:
            wandb.log({"epoch_time": epoch_end_time - epoch_start_time})

    if config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train the match-prediction model")
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default=None,
        help="Name for the Wandb run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        default=False,
        help="Continue training from the last saved model",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load the model from (default: MODEL_PATH)",
    )
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0)",
    )

    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Run validation every N epochs",
    )
    parser.add_argument(
        "--disable-aux-tasks",
        action="store_true",
        help="Disable auxiliary tasks and train only on win prediction",
    )
    args = parser.parse_args()

    # Initialize configuration
    config = TrainingConfig()
    if args.config:
        config.update_from_json(args.config)

    # Update config with command line arguments
    config.dataset_fraction = args.dataset_fraction
    config.validation_interval = args.validation_interval
    config.aux_tasks_enabled = not args.disable_aux_tasks

    print("Training configuration:")
    print(config)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Set random seeds for reproducibility
    set_random_seeds()
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        train_model(
            args.run_name,
            config,
            continue_training=args.continue_training,
            load_path=args.load_path,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        cleanup()
        raise

    if args.profile:
        profiler.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)

        # Optionally, save profiling results to a file
        ps.dump_stats(DATA_DIR + "train_profile.prof")
        print("Profiling data saved to train_profile.prof")
