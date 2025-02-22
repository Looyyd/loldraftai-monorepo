# utils/match_prediction/model.py

import torch
import torch.nn as nn
import pickle

from utils.match_prediction import ENCODERS_PATH
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from utils.match_prediction.config import TrainingConfig


class SimpleMatchModel(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        embed_dim,
        hidden_dims,
        dropout,
    ):
        super(SimpleMatchModel, self).__init__()

        # Store dimensions for debugging
        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for col in CATEGORICAL_COLUMNS:
            self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Champion embeddings
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)

        # Project numerical features
        self.numerical_projection = (
            nn.Linear(len(NUMERICAL_COLUMNS), embed_dim) if NUMERICAL_COLUMNS else None
        )

        # Calculate total input dimension for MLP
        num_categorical = len(CATEGORICAL_COLUMNS)  # Categorical features
        num_champions = len(POSITIONS) * 2  # Champions from both teams
        num_numerical_projections = (
            1 if NUMERICAL_COLUMNS else 0
        )  # Projected numerical features
        total_embed_features = (
            num_categorical + num_champions + num_numerical_projections
        )
        mlp_input_dim = total_embed_features * embed_dim

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions}")
        print(f"- Numerical features projection: {num_numerical_projections}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # MLP layers
        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layers for each task
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)

    def forward(self, features):
        batch_size = features["champion_ids"].size(0)
        embeddings_list = []

        # Process categorical features
        for col in CATEGORICAL_COLUMNS:
            embed = self.embeddings[col](features[col])  # [batch_size, embed_dim]
            embeddings_list.append(embed)

        # Process champion features
        champion_ids = features["champion_ids"]  # [batch_size, num_champions]
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # [batch_size, num_champions, embed_dim]

        # Combine champion and champion other champion features(empty for now because we don't have role percentages or other features)
        champion_features = champion_embeds

        # Reshape champion features to [batch_size, num_champions * embed_dim]
        champion_features = champion_features.view(batch_size, -1)
        embeddings_list.append(champion_features)

        # Process numerical features if they exist
        if self.numerical_projection is not None and NUMERICAL_COLUMNS:
            numerical_features = torch.stack(
                [features[col] for col in NUMERICAL_COLUMNS], dim=1
            )
            numerical_embed = self.numerical_projection(
                numerical_features
            )  # [batch_size, embed_dim]
            embeddings_list.append(numerical_embed)

        # Concatenate all features
        combined_features = torch.cat(embeddings_list, dim=1)

        # Debug print dimensions
        if combined_features.shape[1] != self.mlp[0].weight.shape[1]:
            print(f"\nDimension mismatch!")
            print(f"Combined features shape: {combined_features.shape}")
            print(f"First MLP layer input dim: {self.mlp[0].weight.shape[1]}")
            print(f"First MLP layer weight shape: {self.mlp[0].weight.shape}")
            for i, embedding in enumerate(embeddings_list):
                print(f"Embedding {i} shape: {embedding.shape}")

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs for each task
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs


if __name__ == "__main__":
    # Load configuration
    config = TrainingConfig()

    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    num_champions = 200

    model = SimpleMatchModel(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=config.embed_dim,
        dropout=config.dropout,
    )
    # Print model architecture
    print(model)

    # Calculate and print model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"\nModel Size: {size_all_mb:.3f} MB")

    # Print sizes of individual layers
    print("\nLayer Sizes:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            layer_size = (
                sum(p.nelement() * p.element_size() for p in module.parameters())
                / 1024**2
            )
            print(f"{name}: {layer_size:.3f} MB")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")
