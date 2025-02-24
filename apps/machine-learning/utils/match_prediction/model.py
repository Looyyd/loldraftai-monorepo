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


# Define SwiGLU activation
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # SwiGLU splits the input dimension into two parts
        self.linear = nn.Linear(dim, dim * 2)  # Double the dimension for gate and value

    def forward(self, x):
        # Split the doubled dimension into value and gate
        x = self.linear(x)
        v, g = x.chunk(2, dim=-1)
        # Swish activation: x * sigmoid(x)
        gate = g * torch.sigmoid(g)
        return v * gate


class Model(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        embed_dim,
        hidden_dims,
        dropout,
    ):
        super(Model, self).__init__()

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

        # Calculate total input dimension
        num_categorical = len(CATEGORICAL_COLUMNS)
        num_champions = len(POSITIONS) * 2
        num_numerical_projections = 1 if NUMERICAL_COLUMNS else 0
        total_embed_features = (
            num_categorical + num_champions + num_numerical_projections
        )
        mlp_input_dim = total_embed_features * embed_dim

        # Add positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, total_embed_features, embed_dim)
        )

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions}")
        print(f"- Numerical features projection: {num_numerical_projections}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # Lightweight attention layer for feature interaction
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # MLP with residual connections
        layers = []
        prev_dim = mlp_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(SwiGLU(hidden_dim))  # Replace GELU with SwiGLU
            layers.append(
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else 0.1)
            )  # Lower dropout for final layer
            # Add skip connection if dimensions match
            if i > 0 and hidden_dims[i - 1] == hidden_dim:
                layers.append(ResidualConnection())
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
            embed = self.embeddings[col](features[col])
            embeddings_list.append(embed)

        # Process champion features
        champion_ids = features["champion_ids"]
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # [batch_size, num_champions, embed_dim]
        champion_features = champion_embeds  # Placeholder for future role features
        embeddings_list.append(champion_features.view(batch_size, -1))

        # Process numerical features
        if self.numerical_projection is not None and NUMERICAL_COLUMNS:
            numerical_features = torch.stack(
                [features[col] for col in NUMERICAL_COLUMNS], dim=1
            )
            numerical_embed = self.numerical_projection(numerical_features)
            embeddings_list.append(numerical_embed)

        # Concatenate embeddings and apply attention
        combined_features = torch.cat(
            embeddings_list, dim=1
        )  # [batch_size, total_embed_features * embed_dim]
        combined_features = combined_features.view(
            batch_size, -1, self.embed_dim
        )  # [batch_size, seq_len, embed_dim]

        # Add positional embeddings
        combined_features = (
            combined_features + self.pos_embedding
        )  # [batch_size, seq_len, embed_dim]

        attn_output, _ = self.attention(
            combined_features, combined_features, combined_features
        )
        attn_output = self.attn_norm(attn_output)
        combined_features = attn_output.view(batch_size, -1)  # Flatten back

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs


class ResidualConnection(nn.Module):
    def forward(self, x):
        return x + self.prev_x if hasattr(self, "prev_x") else x

    def forward_pre(self, x):
        self.prev_x = x
        return x


if __name__ == "__main__":
    config = TrainingConfig()

    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    num_champions = 200

    model = Model(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=config.embed_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    print(model)

    param_size = (
        sum(param.nelement() * param.element_size() for param in model.parameters())
        / 1024**2
    )
    print(f"\nModel Size: {param_size:.3f} MB")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
