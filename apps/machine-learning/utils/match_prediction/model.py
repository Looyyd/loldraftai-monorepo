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


class PatchModulatedChampions(nn.Module):
    def __init__(
        self,
        num_champions: int,
        num_patches: int,
        embed_dim: int,
        modulation_scale: float = 0.2,
    ):
        super().__init__()
        # Base champion embedding (unchanged from original)
        self.champion_base = nn.Embedding(num_champions, embed_dim)

        # Patch-specific modulation for each champion
        # We use a smaller scale for modulation to ensure it's minimal by default
        self.modulation_scale = modulation_scale
        self.patch_modulation = nn.Embedding(num_patches * num_champions, embed_dim)

        # Initialize modulation embeddings to small values
        # This ensures they start with minimal effect
        nn.init.normal_(self.patch_modulation.weight, mean=0.0, std=0.02)

    def forward(
        self, champion_ids: torch.Tensor, patch_ids: torch.Tensor
    ) -> torch.Tensor:
        # Get base champion embeddings
        # Shape: [batch_size, num_positions, embed_dim]
        base_embeds = self.champion_base(champion_ids)

        # Create combined patch-champion indices
        # This creates a unique index for each (champion, patch) pair
        batch_size, num_positions = champion_ids.shape
        patch_expanded = patch_ids.unsqueeze(1).expand(-1, num_positions)
        combined_indices = (
            patch_expanded * self.champion_base.num_embeddings + champion_ids
        )

        # Get modulation values
        # Shape: [batch_size, num_positions, embed_dim]
        modulations = self.patch_modulation(combined_indices)

        # Apply modulation as a multiplicative factor
        # Using tanh ensures output is in range [-1, 1], then scaled to be smaller
        # The 1 + modulation structure means by default we stay close to original embedding
        modulated_embeds = base_embeds * (
            1 + self.modulation_scale * torch.tanh(modulations)
        )

        return modulated_embeds


class Model(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        num_patches,
        embed_dim=64,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ):
        super(Model, self).__init__()

        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for col in CATEGORICAL_COLUMNS:
            self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Modified champion embeddings with patch awareness
        self.champion_embedding = PatchModulatedChampions(
            num_champions=num_champions,
            num_patches=num_patches,
            embed_dim=embed_dim,
            modulation_scale=0.2,  # Controls how much patch can affect champions
        )

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

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Champion positions: {num_champions}")
        print(f"- Numerical features projection: {num_numerical_projections}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # Simple but effective MLP with modern components
        layers = []
        prev_dim = mlp_input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer (no bias when followed by BatchNorm)
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))

            # BatchNorm - more stable than LayerNorm for this architecture
            layers.append(nn.BatchNorm1d(hidden_dim))

            # GELU activation - smoother than ReLU but still stable
            layers.append(nn.GELU())

            # Dropout - reduced for later layers
            dropout_rate = dropout if i < len(hidden_dims) - 1 else dropout * 0.5
            layers.append(nn.Dropout(dropout_rate))

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

        # Process champion features with patch awareness
        champion_ids = features["champion_ids"]
        patch_ids = features["numerical_patch"].long()  # Convert to long indices

        # Get patch-modulated champion embeddings
        champion_embeds = self.champion_embedding(champion_ids, patch_ids)

        # Reshape champion embeddings for the MLP
        champion_embeds_flat = champion_embeds.reshape(batch_size, -1)
        embeddings_list.append(champion_embeds_flat)

        # Process numerical features
        if self.numerical_projection is not None and NUMERICAL_COLUMNS:
            numerical_features = torch.stack(
                [features[col] for col in NUMERICAL_COLUMNS], dim=1
            )
            numerical_embed = self.numerical_projection(numerical_features)
            embeddings_list.append(numerical_embed)

        # Concatenate all features
        combined_features = torch.cat(embeddings_list, dim=1)

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs
