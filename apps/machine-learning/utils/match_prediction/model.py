import torch
import torch.nn as nn
import pickle
import math

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


# Residual connection module
class ResidualConnection(nn.Module):
    def forward(self, x):
        return x + self.prev_x if hasattr(self, "prev_x") else x

    def forward_pre(self, x):
        self.prev_x = x
        return x


# MLP Block with normalization, activation, and residual connection
class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, use_residual=False):
        super().__init__()
        self.use_residual = use_residual and input_dim == output_dim

        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = SwiGLU(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x if self.use_residual else None
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.use_residual:
            x = x + residual

        return x


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

        # Add position-specific projections for role awareness
        # This replaces positional embeddings with explicit role projections
        self.role_projections = nn.ModuleDict()
        for i, position in enumerate(POSITIONS):
            # Blue team positions
            self.role_projections[f"blue_{position}"] = nn.Linear(embed_dim, embed_dim)
            # Red team positions
            self.role_projections[f"red_{position}"] = nn.Linear(embed_dim, embed_dim)

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions}")
        print(f"- Numerical features projection: {num_numerical_projections}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # Feature interaction layer - instead of attention, use a feature mixer
        # If numerical features are present, adjust input dimension to include them
        feature_mixer_input_dim = embed_dim * 2 if NUMERICAL_COLUMNS else embed_dim

        self.feature_mixer = nn.Sequential(
            nn.LayerNorm(feature_mixer_input_dim),
            nn.Linear(feature_mixer_input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Lower dropout for mixer
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # MLP blocks with residual connections
        self.mlp_blocks = nn.ModuleList()
        prev_dim = mlp_input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Use residual connections where dimensions match after the first layer
            use_residual = i > 0 and hidden_dims[i - 1] == hidden_dim

            self.mlp_blocks.append(
                MLPBlock(
                    input_dim=prev_dim,
                    output_dim=hidden_dim,
                    dropout=dropout if i < len(hidden_dims) - 1 else dropout * 0.5,
                    use_residual=use_residual,
                )
            )
            prev_dim = hidden_dim

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

        # Process champion features with position-specific projections
        champion_ids = features["champion_ids"]  # [batch_size, num_positions]
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # [batch_size, num_positions, embed_dim]

        # Process numerical features first to condition the feature mixer
        numerical_context = None
        if self.numerical_projection is not None and NUMERICAL_COLUMNS:
            numerical_features = torch.stack(
                [features[col] for col in NUMERICAL_COLUMNS], dim=1
            )
            numerical_context = self.numerical_projection(numerical_features)
            # We'll append this to embeddings_list later

        # Apply position-specific projections for role awareness
        position_aware_embeds = []

        # Blue team positions (first 5)
        for i, position in enumerate(POSITIONS):
            position_embed = self.role_projections[f"blue_{position}"](
                champion_embeds[:, i]
            )
            position_aware_embeds.append(position_embed)

        # Red team positions (last 5)
        for i, position in enumerate(POSITIONS):
            position_embed = self.role_projections[f"red_{position}"](
                champion_embeds[:, i + 5]
            )
            position_aware_embeds.append(position_embed)

        # Process each position embedding through feature mixer with numerical context
        mixed_embeds = []
        for i, embed in enumerate(position_aware_embeds):
            # If we have numerical features, concatenate them with each position embedding
            if numerical_context is not None:
                # Concatenate champion embed with numerical context
                enriched_embed = torch.cat([embed, numerical_context], dim=1)
                # Apply feature mixer with numerical context
                mixed = self.feature_mixer(enriched_embed)
            else:
                # Without numerical features, just use the original mixer
                mixed = self.feature_mixer(embed)

            mixed_embeds.append(mixed)

        # Concatenate all champion embeddings
        champion_features = torch.cat([e.unsqueeze(1) for e in mixed_embeds], dim=1)
        embeddings_list.append(champion_features.view(batch_size, -1))

        # Add numerical features to the overall feature set if not already used
        if numerical_context is not None and self.numerical_projection is not None:
            embeddings_list.append(numerical_context)

        # Numerical features were already processed and added to embeddings_list in the champion processing section

        # Concatenate all embeddings
        combined_features = torch.cat(embeddings_list, dim=1)

        # Pass through MLP blocks
        x = combined_features
        for block in self.mlp_blocks:
            x = block(x)

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs
