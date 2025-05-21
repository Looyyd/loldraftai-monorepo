# utils/match_prediction/train_utils.py

import torch.nn as nn
import torch
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import wandb
from sklearn.preprocessing import LabelEncoder

from utils.match_prediction import (
    get_best_device,
    MODEL_CONFIG_PATH,
    CHAMPION_ID_ENCODER_PATH,
)
from utils.match_prediction.column_definitions import (
    COLUMNS,
    KNOWN_CATEGORICAL_COLUMNS_NAMES,
    possible_values_queue_type,
    ColumnType,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.model import Model
from utils.match_prediction.champions import Champion, ChampionClass


def get_optimizer_grouped_parameters(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    # Get all parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Separate parameters into decay and no-decay groups
    decay_params = []
    nodecay_params = []

    for name, param in param_dict.items():
        # No weight decay for:
        # 1. Standard embedding layers (embeddings.*.weight and champion_embedding.weight)
        # 2. All bias terms
        # 3. All normalization layers
        # source: https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025
        if (
            "embeddings." in name
            or "champion_embedding." in name
            or "patch_embedding." in name
            or "champion_patch_embedding." in name
            or "bias" in name
            or "norm" in name
        ):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    # Create optimizer groups
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Print statistics
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"\nnum decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )

    return optim_groups


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_num_champions() -> tuple[int, int]:
    """Get the total number of champion embeddings needed and the ID for unknown champions.

    Returns:
        tuple[int, int]: (num_champions, unknown_champion_id)
    """
    with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
        champion_id_mapping = pickle.load(f)["mapping"]

    # Get the unknown champion ID directly from the encoder
    unknown_champion_id = champion_id_mapping.transform(["UNKNOWN"])[0]
    # Total number of embeddings is simply the number of classes in the encoder
    num_champions = len(champion_id_mapping.classes_)

    return num_champions, unknown_champion_id


def collate_fn(
    batch: List[Dict[str, Any]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # Convert columns to tensors directly
    collated = {
        "champion_ids": torch.stack([item["champion_ids"] for item in batch]),
        "patch": torch.tensor([item["patch"] for item in batch], dtype=torch.long),
    }

    # Handle other categorical columns
    for col, col_def in COLUMNS.items():
        if (
            col not in ("champion_ids", "patch")
            and col_def.column_type == ColumnType.KNOWN_CATEGORICAL
        ):
            collated[col] = torch.tensor(
                [item[col] for item in batch], dtype=torch.long
            )

    # Convert labels to tensors directly
    collated_labels = {
        task_name: torch.tensor([item[task_name] for item in batch], dtype=torch.float)
        # Only collate labels that actually exist in the input data
        # TODO: this is only because of masked losses, maybe could be refactored to only ignore those
        for task_name in TASKS
        if task_name in batch[0]
    }

    return collated, collated_labels


def init_model(
    config: TrainingConfig,
    num_champions: int,
    continue_training: bool,
    load_path: Optional[str] = None,
) -> Model:
    """Initialize the model with pre-initialized embeddings.

    Args:
        config: Training configuration
        num_champions: Number of champions to support
        continue_training: Whether to continue training from a checkpoint
        load_path: Path to load model from if continue_training is True
        use_custom_init: Whether to use custom initialization for embeddings (default: False)
    """
    # Initialize model
    model = Model(
        config=config,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    if continue_training and load_path and Path(load_path).exists():
        print(f"Loading model from {load_path}")
        state_dict = torch.load(load_path, weights_only=True)
        # Remove '_orig_mod.' prefix from state dict keys if present
        fixed_state_dict = {
            k.replace("_orig_mod.", ""): state_dict[k] for k in state_dict.keys()
        }
        model.load_state_dict(fixed_state_dict)

    model_params = {
        **config.to_dict(),
    }

    with open(MODEL_CONFIG_PATH, "wb") as f:
        pickle.dump(model_params, f)

    device = get_best_device()
    model.to(device)
    if device == torch.device("cuda"):
        print("Compiling model")
        model = torch.compile(model, backend="eager")
        print("Model compiled")

    if config.log_wandb:
        wandb.watch(model, log_freq=1000)

    return model
