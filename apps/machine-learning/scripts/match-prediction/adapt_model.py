# This script is used to adapt the model for online learning.
# More specifically, it adapts the model for a new patch, by either sliding the embeddings(if we want the total number of patches to remain the same)
# or by adding new patch embeddings(if we want to add a new patch).
import torch
import torch.nn as nn
import argparse
import pickle
from utils.match_prediction import (
    MODEL_PATH,
    PATCH_MAPPING_PATH,
    CHAMPION_ID_ENCODER_PATH,
)
from utils.match_prediction.config import TrainingConfig
from utils.match_prediction.model import (
    Model,
)


def load_model(config: TrainingConfig):
    """Initialize a new model with the current configuration."""
    return Model(config, hidden_dims=config.hidden_dims, dropout=config.dropout)


def save_model(model: Model, model_path: str):
    """Save the updated model to the specified path."""
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def slide_embeddings(model: Model, old_num_patches: int):
    """Slide embeddings when the number of patches remains the same."""
    with torch.no_grad():
        # Ensure we don’t exceed the old model’s embeddings
        weights_to_shift = min(old_num_patches, model.num_patches)
        # Shift patch embeddings: 1->0, 2->1, ..., (N-1)->(N-2), (N-1) initialized with (N-2)
        model.patch_embedding.weight[: weights_to_shift - 1] = (
            model.patch_embedding.weight[1:weights_to_shift].clone()
        )
        if weights_to_shift > 1:
            model.patch_embedding.weight[weights_to_shift - 1] = (
                model.patch_embedding.weight[weights_to_shift - 2].clone()
            )

        # Shift champion_patch embeddings for each champion
        num_champions = model.num_champions
        num_patches = model.num_patches
        for c in range(num_champions):
            start_idx = c * num_patches
            old_start_idx = c * old_num_patches
            # Shift within the old number of patches
            if (
                old_start_idx + weights_to_shift
                <= model.champion_patch_embedding.weight.size(0)
            ):
                model.champion_patch_embedding.weight[
                    start_idx : start_idx + weights_to_shift - 1
                ] = model.champion_patch_embedding.weight[
                    start_idx + 1 : start_idx + weights_to_shift
                ].clone()
                if weights_to_shift > 1:
                    model.champion_patch_embedding.weight[
                        start_idx + weights_to_shift - 1
                    ] = model.champion_patch_embedding.weight[
                        start_idx + weights_to_shift - 2
                    ].clone()


def modify_state_dict_for_new_patch(
    state_dict, old_num_patches, new_num_patches, num_champions
):
    """Modify the state dictionary to accommodate a new patch."""
    modified_state_dict = state_dict.copy()

    # Modify patch_embedding.weight
    old_patch_weight = state_dict[
        "patch_embedding.weight"
    ]  # Shape: [old_num_patches, embed_dim]
    embed_dim = old_patch_weight.size(1)
    new_patch_weight = torch.zeros(
        new_num_patches, embed_dim, dtype=old_patch_weight.dtype
    )
    new_patch_weight[:old_num_patches] = old_patch_weight
    # TODO: could experiment with using default initialization, but the idea is that using the last patch could avoid overfitting to the last patch, if it has less data
    # Initialize new patch embeddings with the last old patch embedding
    if old_num_patches > 0:
        last_patch_weight = old_patch_weight[old_num_patches - 1]
        new_patch_weight[old_num_patches:] = last_patch_weight.unsqueeze(0).expand(
            new_num_patches - old_num_patches, -1
        )
    modified_state_dict["patch_embedding.weight"] = new_patch_weight

    # Modify champion_patch_embedding.weight
    old_champ_patch_weight = state_dict[
        "champion_patch_embedding.weight"
    ]  # Shape: [num_champions * old_num_patches, embed_dim]
    champ_patch_embed_dim = old_champ_patch_weight.size(1)
    new_champ_patch_weight = torch.zeros(
        num_champions * new_num_patches,
        champ_patch_embed_dim,
        dtype=old_champ_patch_weight.dtype,
    )
    for c in range(num_champions):
        old_start_idx = c * old_num_patches
        new_start_idx = c * new_num_patches
        # Copy existing champion-patch embeddings
        new_champ_patch_weight[new_start_idx : new_start_idx + old_num_patches] = (
            old_champ_patch_weight[old_start_idx : old_start_idx + old_num_patches]
        )
        # Initialize new patch embeddings with the last old patch embedding for this champion
        if old_num_patches > 0:
            last_old_idx = old_start_idx + old_num_patches - 1
            last_patch_weight = old_champ_patch_weight[last_old_idx]
            new_champ_patch_weight[
                new_start_idx + old_num_patches : new_start_idx + new_num_patches
            ] = last_patch_weight.unsqueeze(0).expand(
                new_num_patches - old_num_patches, -1
            )
    modified_state_dict["champion_patch_embedding.weight"] = new_champ_patch_weight

    return modified_state_dict


def main():
    parser = argparse.ArgumentParser(description="Adapt model for online learning")
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["slide", "add"],
        required=True,
        help="Action to perform: 'slide' for sliding embeddings, 'add' for adding a new patch",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the updated model (default: overwrite input model)",
    )
    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig()

    # Load current patch mapping to determine new_num_patches
    with open(PATCH_MAPPING_PATH, "rb") as f:
        patch_mapping = pickle.load(f)["mapping"]
    new_num_patches = len(patch_mapping)

    # Load pre-trained state_dict
    state_dict = torch.load(args.model_path, weights_only=True)
    # Remove '_orig_mod.' prefix from state dict keys if present
    fixed_state_dict = {
        k.replace("_orig_mod.", ""): state_dict[k] for k in state_dict.keys()
    }

    # Determine old_num_patches from state_dict
    old_patch_weight = fixed_state_dict["patch_embedding.weight"]
    old_num_patches = old_patch_weight.size(0)

    # Initialize the model with current configuration (reflects current patch mapping)
    model = load_model(config)

    # Get num_champions from the model
    num_champions = model.num_champions

    if args.action == "slide":
        if new_num_patches != old_num_patches:
            print(
                f"Warning: 'slide' action with new_num_patches ({new_num_patches}) != old_num_patches ({old_num_patches}). Proceeding with shift."
            )
        # Load state_dict first
        model.load_state_dict(fixed_state_dict, strict=False)
        slide_embeddings(model, old_num_patches)
        print("Embeddings slid successfully.")
    elif args.action == "add":
        if new_num_patches <= old_num_patches:
            print(
                f"Warning: 'add' action but new_num_patches ({new_num_patches}) <= old_num_patches ({old_num_patches}). Proceeding with initialization."
            )
        # Modify state_dict to match new model shapes
        modified_state_dict = modify_state_dict_for_new_patch(
            fixed_state_dict, old_num_patches, new_num_patches, num_champions
        )
        # Load modified state_dict
        model.load_state_dict(modified_state_dict, strict=False)
        print("New patch embeddings initialized successfully.")

    # Save the updated model
    output_path = args.output_path if args.output_path else args.model_path
    save_model(model, output_path)


if __name__ == "__main__":
    main()
