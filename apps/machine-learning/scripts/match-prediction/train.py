# scripts/match-prediction/train.py
import signal
import copy
import pickle
import datetime
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.data import DataLoader
import argparse

from torch.optim.lr_scheduler import OneCycleLR

from utils.match_prediction.match_dataset import MatchDataset, dataloader_config
from utils.match_prediction.model import Model
from utils.match_prediction import (
    get_best_device,
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_BATCH_SIZE,
    PATCH_MAPPING_PATH,
)
from utils.match_prediction.column_definitions import (
    KNOWN_CATEGORICAL_COLUMNS_NAMES,
)
from utils.match_prediction.task_definitions import TASKS, TaskType, get_enabled_tasks
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.train_utils import (
    get_optimizer_grouped_parameters,
    set_random_seeds,
    get_num_champions,
    collate_fn,
    init_model,
)
from utils.match_prediction.champions import Champion, ChampionClass

# Enable cuDNN auto-tuner, source https://x.com/karpathy/status/1299921324333170689/photo/1
torch.backends.cudnn.benchmark = True
# should use TensorFloat-32, which is faster that "highest" precision
torch.set_float32_matmul_precision("high")

device = get_best_device()


def save_model(model: Model, timestamp: Optional[str] = None) -> str:
    if config.debug:
        print("Debug mode: Skipping model save")
        return "debug_mode_no_save"

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_timestamp_path = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pth"
    torch.save(model.state_dict(), model_timestamp_path)
    print(f"Model saved to {model_timestamp_path}")
    return model_timestamp_path


def cleanup() -> None:
    if "model" in globals() and not config.debug:
        save_model(model)
    if wandb.run is not None:
        wandb.finish()


def signal_handler(signum: int, frame: Any) -> None:
    print("Received interrupt signal. Saving model and exiting...")
    cleanup()
    exit(0)


def apply_label_smoothing(labels: torch.Tensor, smoothing: float = 0.2) -> torch.Tensor:
    # Apply label smoothing
    return labels * (1 - smoothing) + 0.5 * smoothing


def train_epoch(
    model: Model,
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
    enabled_tasks = get_enabled_tasks(config, epoch)
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

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(features)
            losses = torch.stack(
                [
                    criterion[task_name](outputs[task_name], labels[task_name])
                    for task_name in task_names
                ]
            )
            task_loss = (losses * task_weights).sum()

            # Patch regularization
            patch_reg_loss = torch.tensor(0.0, device=device)
            if model.num_patches > 1:
                for i in range(model.num_patches - 1):
                    diff = (
                        model.patch_embedding.weight[i]
                        - model.patch_embedding.weight[i + 1]
                    )
                    patch_reg_loss += torch.norm(diff, p=2)
                patch_reg_loss /= model.num_patches - 1

            # Champion+patch regularization
            champ_patch_reg_loss = torch.tensor(0.0, device=device)
            if model.num_patches > 1:
                for c in range(model.num_champions):
                    for p in range(model.num_patches - 1):
                        idx1 = c * model.num_patches + p
                        idx2 = c * model.num_patches + p + 1
                        diff = (
                            model.champion_patch_embedding.weight[idx1]
                            - model.champion_patch_embedding.weight[idx2]
                        )
                        champ_patch_reg_loss += torch.norm(diff, p=2)
                champ_patch_reg_loss /= model.num_champions * (model.num_patches - 1)

            # Combine losses with regularization weights
            total_loss = (
                task_loss
                + config.patch_reg_lambda * patch_reg_loss
                + config.champ_patch_reg_lambda * champ_patch_reg_loss
            ) / config.accumulation_steps

        total_loss.backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler is not None and batch_idx < len(train_loader) - 1:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # log metrics every 20 steps
            if (
                (batch_idx + 1) // config.accumulation_steps
            ) % 20 == 0 and config.log_wandb:
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler
                    else optimizer.param_groups[0]["lr"]
                )
                log_training_step(
                    epoch,
                    batch_idx,
                    grad_norm,
                    losses,
                    task_names,
                    config,
                    current_lr,
                    patch_reg_loss.item(),
                    champ_patch_reg_loss.item(),
                    model,
                )

        epoch_loss += total_loss.item()
        epoch_steps += 1

    return epoch_loss, epoch_steps


def log_training_step(
    epoch: int,
    batch_idx: int,
    grad_norm: float,
    losses: torch.Tensor,
    task_names: List[str],
    config: TrainingConfig,
    current_lr: float,
    patch_reg_loss: float,
    champ_patch_reg_loss: float,
    model: Model,
) -> None:
    if config.log_wandb:
        # Calculate L2 norms for different parameter groups
        categorical_norm = torch.norm(
            torch.cat(
                [p.data.view(-1) for name, p in model.embeddings.named_parameters()]
            )
        ).item()
        patch_norm = torch.norm(model.patch_embedding.weight.data).item()
        champ_patch_norm = torch.norm(model.champion_patch_embedding.weight.data).item()

        mlp_norm = torch.norm(
            torch.cat([p.data.view(-1) for p in model.mlp.parameters()])
        ).item()
        output_norm = torch.norm(
            torch.cat([p.data.view(-1) for p in model.output_layers.parameters()])
        ).item()

        log_data = {
            "epoch": epoch + 1,
            "batch": (batch_idx + 1) // config.accumulation_steps,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "patch_reg_loss": patch_reg_loss,
            "champ_patch_reg_loss": champ_patch_reg_loss,
            # Parameter norms
            "categorical_embed_norm": categorical_norm,
            "patch_embed_norm": patch_norm,
            "champ_patch_embed_norm": champ_patch_norm,
            "mlp_norm": mlp_norm,
            "output_layers_norm": output_norm,
        }
        log_data.update(
            {f"train_loss_{k}": v.item() for k, v in zip(task_names, losses)}
        )
        wandb.log(log_data)


def validate(
    model: Model,
    test_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    epoch: int,
) -> Tuple[Optional[float], Dict[str, float]]:
    model.eval()
    enabled_tasks = get_enabled_tasks(config, epoch)

    # Initialize accumulators on the device (GPU/MPS) to avoid CPU-GPU transfers
    loss_accumulators = {
        task_name: torch.zeros(2, device=device) for task_name in enabled_tasks.keys()
    }

    # Add accuracy accumulator for win_prediction
    win_prediction_accuracy = (
        torch.zeros(2, device=device) if "win_prediction" in enabled_tasks else None
    )

    # Load patch mapping and numerical stats for denormalization
    try:
        with open(PATCH_MAPPING_PATH, "rb") as f:
            patch_data = pickle.load(f)
            reverse_patch_mapping = {v: k for k, v in patch_data["mapping"].items()}
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load patch mapping or stats: {e}")
        reverse_patch_mapping = None

    # Subset accumulators for win prediction (on GPU)
    if config.track_subset_val_losses and "win_prediction" in enabled_tasks:
        patch_losses = {}  # {patch_val: [loss_sum, count]}
        elo_losses = {}  # {elo_val: [loss_sum, count]}
        queue_losses = {}  # {queue_id: [loss_sum, count]}

    total_loss = torch.tensor(0.0, device=device)
    total_steps = 0
    task_weights = torch.tensor(
        [task_def.weight for task_def in enabled_tasks.values()], device=device
    )

    with torch.no_grad():
        for features, labels in test_loader:
            features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
            batch_size = next(iter(labels.values())).size(0)

            with torch.autocast(
                device_type=device.type if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
            ):
                outputs = model(features)
                if config.calculate_val_loss:
                    # Calculate all task losses once using functional versions
                    task_losses = {}
                    for task_name, task_def in enabled_tasks.items():
                        if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                            loss = nn.functional.binary_cross_entropy_with_logits(
                                outputs[task_name], labels[task_name], reduction="none"
                            )
                        elif task_def.task_type == TaskType.REGRESSION:
                            loss = nn.functional.mse_loss(
                                outputs[task_name], labels[task_name], reduction="none"
                            )
                        task_losses[task_name] = loss

                    # Update loss accumulators
                    for task_name, loss in task_losses.items():
                        loss_sum = loss.sum()
                        loss_accumulators[task_name][0] += loss_sum
                        loss_accumulators[task_name][1] += batch_size

                    # Calculate total weighted loss
                    losses = torch.stack([loss.mean() for loss in task_losses.values()])
                    total_loss += (losses * task_weights).sum()

                    # Calculate win prediction accuracy
                    if "win_prediction" in enabled_tasks:
                        # Apply sigmoid to get probabilities
                        win_probs = torch.sigmoid(outputs["win_prediction"])
                        # Predicted win if probability > 0.5
                        predictions = (win_probs > 0.5).float()
                        # Count correct predictions
                        correct = (predictions == labels["win_prediction"]).sum()
                        win_prediction_accuracy[0] += correct
                        win_prediction_accuracy[1] += batch_size

                    # Track subset losses if needed
                    if (
                        config.track_subset_val_losses
                        and "win_prediction" in enabled_tasks
                    ):
                        win_pred_losses = task_losses["win_prediction"]
                        for key, values in [
                            ("patch", features["patch"]),
                            ("elo", features["elo"]),
                            ("queue", features["queue_type"]),
                        ]:
                            unique_vals, inverse = torch.unique(
                                values, return_inverse=True
                            )
                            counts = torch.bincount(inverse)
                            sums = torch.zeros_like(
                                unique_vals, dtype=torch.float, device=device
                            )
                            sums.scatter_add_(0, inverse, win_pred_losses)
                            current_metrics = locals()[f"{key}_losses"]
                            for val, sum_val, count in zip(unique_vals, sums, counts):
                                val_item = val.item()
                                if val_item not in current_metrics:
                                    current_metrics[val_item] = [0.0, 0]
                                current_metrics[val_item][0] += sum_val.item()
                                current_metrics[val_item][1] += count.item()

            total_steps += 1

    # Compute final metrics and transfer only the result to CPU
    losses = calculate_final_loss(loss_accumulators)
    avg_loss = (total_loss / total_steps).item() if config.calculate_val_loss else None

    # Add win prediction accuracy to metrics
    if win_prediction_accuracy is not None:
        accuracy = (win_prediction_accuracy[0] / win_prediction_accuracy[1]).item()
        losses["win_prediction_accuracy"] = accuracy

    # Add subset metrics to the metrics dictionary
    if config.track_subset_val_losses and "win_prediction" in enabled_tasks:
        # Patch metrics
        for patch_val, (loss_sum, count) in patch_losses.items():
            if reverse_patch_mapping:
                original_patch = reverse_patch_mapping.get(patch_val)
                losses[f"win_prediction_patch_{original_patch}"] = loss_sum / count

        # ELO metrics
        for elo_val, (loss_sum, count) in elo_losses.items():
            losses[f"win_prediction_elo_{elo_val}"] = loss_sum / count

        # Queue metrics
        for queue_val, (loss_sum, count) in queue_losses.items():
            losses[f"win_prediction_queue_{queue_val}"] = loss_sum / count

    return avg_loss, losses


def calculate_final_loss(
    loss_accumulators: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    losses = {}
    for task_name, accumulator in loss_accumulators.items():
        # Calculate average loss for all task types
        losses[task_name] = (accumulator[0] / accumulator[1]).item()
    return losses


def log_validation_metrics(
    val_metrics: Dict[str, float], config: TrainingConfig
) -> None:
    if config.log_wandb:
        log_data = {}
        for task_name, metric_value in val_metrics.items():
            if (
                config.calculate_val_win_prediction_only
                and task_name != "win_prediction"
                and task_name != "win_prediction_accuracy"  # Allow accuracy metric
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
        config_dict["num_samples"] = len(train_dataset)
        wandb.init(project="draftking", name=run_name, config=config_dict)

    train_loader, test_loader, test_masked_loader = (
        DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            **dataloader_config,
            collate_fn=collate_fn,
            persistent_workers=device.type == "cuda",
        )
        for dataset in [train_dataset, test_dataset, test_masked_dataset]
    )

    model = init_model(config, num_champions, continue_training, load_path)

    criterions = {}

    def update_criterions(epoch: int) -> None:
        """Update criterions based on current enabled tasks"""
        enabled_tasks = get_enabled_tasks(config, epoch)
        criterions.clear()  # Remove old criterions
        for task_name, task_def in enabled_tasks.items():
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                criterions[task_name] = nn.BCEWithLogitsLoss()
            elif task_def.task_type == TaskType.REGRESSION:
                criterions[task_name] = nn.MSELoss()

    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, config.weight_decay),
        lr=config.learning_rate,
        fused=True if device.type == "cuda" else False,
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
        # Update criterions if needed
        if epoch == 0 or epoch == config.annealing_epoch:
            update_criterions(epoch)
        epoch_start_time = time.time()

        epoch_loss, epoch_steps = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler if config.use_one_cycle_lr else None,  # Pass scheduler
            criterions,
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
            val_loss, val_metrics = validate(model, test_loader, config, device, epoch)
            val_masked_loss, val_masked_metrics = validate(
                model, test_masked_loader, config, device, epoch
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
                    if not config.debug:
                        best_model_state = copy.deepcopy(model.state_dict())
                        torch.save(best_model_state, MODEL_PATH)
                        print(
                            f"New best model saved with validation loss: {best_metric:.4f}"
                        )

            log_validation_metrics(val_metrics, config)

        # save with timestamp anyways
        save_model(model)

        epoch_end_time = time.time()
        if config.log_wandb:
            wandb.log({"epoch_time": epoch_end_time - epoch_start_time})

    if config.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train the match-prediction model")
    parser.add_argument(
        "--run-name",
        type=str,
        required=False,
        default=None,
        help="Name for the Wandb run",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        default=False,
        help="Continue training from the last saved model",
    )
    parser.add_argument(
        "--load-path",
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
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode: no model saving, no wandb logging",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = TrainingConfig()

    # Update config with command line arguments
    config.dataset_fraction = args.dataset_fraction
    config.validation_interval = args.validation_interval
    config.debug = args.debug

    # If in debug mode, disable wandb logging
    if config.debug:
        config.log_wandb = False
        print("Running in DEBUG mode: No model saving, no wandb logging")

    print("Training configuration:")
    print(config)

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
