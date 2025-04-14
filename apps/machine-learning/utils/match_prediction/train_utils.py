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
            or "pos_embedding" in name
            or "pos_scale" in name
            or "numerical_projection" in name
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


def get_num_champions() -> tuple[int, Dict[str, int]]:
    """Get the total number of champion embeddings needed and the IDs for unknown champions per role.

    Returns:
        tuple[int, Dict[str, int]]: (num_champions, {role: unknown_champion_id})
    """
    with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
        champion_id_mapping = pickle.load(f)["mapping"]

    # Get the unknown champion IDs for each role
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    unknown_champion_ids = {
        role: champion_id_mapping.transform([f"UNKNOWN_{role}"])[0] for role in roles
    }

    # Total number of embeddings is the number of classes in the encoder
    num_champions = len(champion_id_mapping.classes_)

    return num_champions, unknown_champion_ids


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


def _initialize_queue_embeddings(embed_dim: int) -> torch.Tensor:
    """Initialize queue type embeddings with a hybrid approach.

    The embeddings are split into two halves:
    - First half: Random initialization for diversity
    - Second half: Biased initialization where all queue types share a similar base vector
    with small noise, encouraging some commonality between queue types

    Half random init is to have some ability to dinstinguish between champions,
    inspired by the paper:
    The Unreasonable Effectiveness of Random Target Embeddings for Continuous-Output Neural Machine Translation
    https://arxiv.org/abs/2310.20620
    """
    half_dim = embed_dim // 2
    queue_embeddings = (
        torch.randn(len(possible_values_queue_type), embed_dim) * 0.02
    )  # Default random init

    # Only modify second half with biased initialization
    queue_base_vector = torch.randn(half_dim) * 0.1
    print("\nQueue Embedding Initialization:")
    for queue_id in possible_values_queue_type:
        noise = torch.randn(half_dim) * 0.01
        queue_embeddings[queue_id, half_dim:] = queue_base_vector + noise
        print(f"Queue {queue_id}: initialized with split random/biased")

    return queue_embeddings


def _initialize_champion_embeddings(
    num_champions: int,
    embed_dim: int,
    champion_encoder,
    champ_to_class: dict,
    champ_display_names: dict,
) -> tuple[torch.Tensor, dict[ChampionClass, int], list[str]]:
    """
    Initialize champion embeddings using class-based biased initialization.

    The embeddings are split into two halves:
    - First half: Random initialization for champion-specific features
    - Second half: Class-based initialization where champions of the same class
    (e.g., MAGE, TANK) share a similar base vector with small noise, encouraging
    champions of the same class to have similar representations

    Half random init is to have some ability to dinstinguish between champions,
    inspired by the paper:
    The Unreasonable Effectiveness of Random Target Embeddings for Continuous-Output Neural Machine Translation
    https://arxiv.org/abs/2310.20620
    """
    half_dim = embed_dim // 2
    # Initialize all embeddings with random values
    base_embeddings = torch.randn(num_champions, embed_dim) * 0.02

    # Create class means for the biased half
    class_means = {
        class_type: torch.randn(half_dim) * 0.1
        for class_type in [
            ChampionClass.MAGE,
            ChampionClass.TANK,
            ChampionClass.BRUISER,
            ChampionClass.ASSASSIN,
            ChampionClass.ADC,
            ChampionClass.ENCHANTER,
        ]
    }

    class_counts = {class_type: 0 for class_type in ChampionClass}
    missing_class_names = []

    # Only modify second half with class-based initialization
    for raw_id in champion_encoder.classes_:
        idx = champion_encoder.transform([raw_id])[0]
        if str(raw_id) == "UNKNOWN":
            base_embeddings[idx, half_dim:] = torch.zeros(half_dim)
            continue

        champ_class = champ_to_class.get(str(raw_id), ChampionClass.UNIQUE)
        if champ_class != ChampionClass.UNIQUE:
            mean = class_means[champ_class]
            noise = torch.randn(half_dim) * 0.01
            base_embeddings[idx, half_dim:] = mean + noise

        class_counts[champ_class] += 1
        if champ_class == ChampionClass.UNIQUE:
            display_name = champ_display_names.get(str(raw_id), f"Champion {raw_id}")
            missing_class_names.append(display_name)

    return base_embeddings, class_counts, missing_class_names


def _log_initialization_stats(
    num_champions: int,
    class_counts: dict[ChampionClass, int],
    missing_class_names: list[str],
    config: TrainingConfig,
) -> None:
    """Log initialization statistics to console and wandb if enabled."""
    print("\nChampion Embedding Initialization Statistics:")
    print(f"Total champions: {num_champions}")
    for class_type, count in class_counts.items():
        print(f"{class_type.name}: {count} champions")
    if missing_class_names:
        print(
            f"\nWarning: {len(missing_class_names)} champions without class assignment:"
        )
        for name in sorted(missing_class_names):
            print(f"  - {name}")
    print()

    if config.log_wandb:
        wandb.log(
            {
                "init_total_champions": num_champions,
                **{
                    f"init_{class_type.name}_count": count
                    for class_type, count in class_counts.items()
                },
                "init_missing_classes_count": len(missing_class_names),
            }
        )


def init_model(
    config: TrainingConfig,
    num_champions: int,
    continue_training: bool,
    load_path: Optional[str] = None,
    use_custom_init: bool = True,
) -> Model:
    """Initialize the model with pre-initialized embeddings.

    Args:
        config: Training configuration
        num_champions: Number of champions to support
        continue_training: Whether to continue training from a checkpoint
        load_path: Path to load model from if continue_training is True
        use_custom_init: Whether to use custom initialization for embeddings (default: False)
    """
    # Load encoders and create mappings
    with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
        champion_id_mapping = pickle.load(f)["mapping"]

    champ_to_class = {str(champ.id): champ.champion_class for champ in Champion}
    champ_display_names = {str(champ.id): champ.display_name for champ in Champion}

    # Initialize model
    model = Model(
        config=config,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    # Apply custom initialization only if requested
    if use_custom_init:
        # Initialize queue embeddings
        queue_embeddings = _initialize_queue_embeddings(
            embed_dim=config.queue_type_embed_dim,
        )
        model.embeddings["queue_type"].weight.data = queue_embeddings

        # Initialize champion embeddings
        champion_embeddings, class_counts, missing_class_names = (
            _initialize_champion_embeddings(
                num_champions=num_champions,
                embed_dim=config.champion_embed_dim,
                champion_encoder=champion_id_mapping,
                champ_to_class=champ_to_class,
                champ_display_names=champ_display_names,
            )
        )
        model.champion_embedding.weight.data = champion_embeddings

        # Initialize champion+patch embeddings (half random, half class-based)
        champion_patch_embeddings = (
            torch.randn(
                num_champions * model.num_patches, config.champion_patch_embed_dim
            )
            * 0.02
        )
        half_dim_champion_patch = config.champion_patch_embed_dim // 2
        for c in range(num_champions):
            base_vector = champion_embeddings[
                c, :half_dim_champion_patch
            ]  # Use first half of champion embedding
            for p in range(model.num_patches):
                idx = c * model.num_patches + p
                noise = torch.randn(half_dim_champion_patch) * 0.01
                champion_patch_embeddings[idx, half_dim_champion_patch:] = (
                    base_vector + noise
                )
        model.champion_patch_embedding.weight.data = champion_patch_embeddings

        _log_initialization_stats(
            num_champions, class_counts, missing_class_names, config
        )

        # Patch embeddings initialization (half random, half similar)
        patch_embeddings = torch.randn(model.num_patches, config.patch_embed_dim) * 0.02
        half_dim_patch = config.patch_embed_dim // 2
        base_patch_vector = torch.randn(half_dim_patch) * 0.1
        for i in range(model.num_patches):
            noise = torch.randn(half_dim_patch) * 0.01
            patch_embeddings[i, half_dim_patch:] = base_patch_vector + noise
        model.patch_embedding.weight.data = patch_embeddings

        # Log embedding statistics
        if config.log_wandb:
            patch_embed_mean = patch_embeddings[:, half_dim_patch:].mean().item()
            patch_embed_std = patch_embeddings[:, half_dim_patch:].std().item()

            # Handle single patch case
            if model.num_patches > 1:
                patch_embed_max_diff = torch.max(
                    torch.pdist(patch_embeddings[:, half_dim_patch:])
                ).item()
            else:
                print(
                    "Warning: Training with single patch, skipping patch distance calculations"
                )
                patch_embed_max_diff = 0.0

            wandb.log(
                {
                    "init_patch_embed_mean": patch_embed_mean,
                    "init_patch_embed_std": patch_embed_std,
                    "init_patch_embed_max_diff": patch_embed_max_diff,
                    "init_champion_embed_mean": champion_embeddings.mean().item(),
                    "init_champion_embed_std": champion_embeddings.std().item(),
                    "init_queue_embed_max_diff": torch.max(
                        torch.pdist(queue_embeddings)
                    ).item(),
                    "init_queue_embed_mean": queue_embeddings.mean().item(),
                    "init_queue_embed_std": queue_embeddings.std().item(),
                    "init_champion_patch_embed_mean": champion_patch_embeddings.mean().item(),
                    "init_champion_patch_embed_std": champion_patch_embeddings.std().item(),
                }
            )
    else:
        print("Using default PyTorch initialization for embeddings")

    if continue_training and load_path and Path(load_path).exists():
        print(f"Loading model from {load_path}")
        model.load_state_dict(torch.load(load_path, weights_only=True))

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
