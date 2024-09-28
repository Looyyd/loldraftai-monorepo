# utils/model.py
import torch
import torch.nn as nn
import pickle

from utils import ENCODERS_PATH
from utils.column_definitions import (
    COLUMNS,
    ColumnType,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    POSITIONS,
)
from utils.task_definitions import TASKS, TaskType


class MatchOutcomeModel(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        num_numerical_features=0,
        embed_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.1,
    ):
        super(MatchOutcomeModel, self).__init__()
        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        self.embed_dim = embed_dim
        self.num_positions = len(POSITIONS) * 2  # Assuming two teams

        for col, col_def in COLUMNS.items():
            if col_def.column_type == ColumnType.CATEGORICAL:
                self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)

        # Embedding for champions
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)
        # Positional embedding for roles
        self.position_embedding = nn.Embedding(self.num_positions, embed_dim)

        # Transformer Encoder for champion embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Linear layer for numerical features
        self.num_numerical_features = num_numerical_features
        if num_numerical_features > 0:
            self.numerical_layer = nn.Sequential(
                nn.Linear(num_numerical_features, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            )

        # Fully connected layers
        total_feature_dim = (
            len(CATEGORICAL_COLUMNS) * embed_dim  # Categorical features
            + embed_dim  # Processed champion features
            + (embed_dim if num_numerical_features > 0 else 0)  # Numerical features
        )

        self.fc = nn.Sequential(
            nn.Linear(total_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Create output layers for each task
        self.output_layers = nn.ModuleDict()

        for task_name, task_def in TASKS.items():
            # Output layers for tasks using shared representation
            if task_def.task_type == TaskType.BINARY_CLASSIFICATION:
                self.output_layers[task_name] = nn.Sequential(
                    nn.Linear(128, 1),
                    nn.Sigmoid(),
                )
            elif task_def.task_type == TaskType.REGRESSION:
                self.output_layers[task_name] = nn.Linear(128, 1)

    def forward(self, features):
        embedded_features = []

        # Process categorical features
        for col in CATEGORICAL_COLUMNS:
            embedded = self.embeddings[col](features[col])  # [batch_size, embed_dim]
            embedded_features.append(embedded)

        # Process numerical features
        if self.num_numerical_features > 0:
            numerical_features = torch.stack(
                [features[col] for col in NUMERICAL_COLUMNS], dim=1
            )
            numerical_embedding = self.numerical_layer(numerical_features)
            embedded_features.append(numerical_embedding)

        # Process champion embeddings with positional embeddings
        # Assume features['champion_ids'] has shape [batch_size, num_champions]
        batch_size = features["champion_ids"].size(0)
        champion_embeds = self.champion_embedding(
            features["champion_ids"]
        )  # [batch_size, num_champions, embed_dim]

        # Create position indices tensor
        position_indices = torch.arange(
            self.num_positions, device=features["champion_ids"].device
        )
        position_indices = position_indices.unsqueeze(0).expand(
            batch_size, -1
        )  # [batch_size, num_champions]
        position_embeds = self.position_embedding(
            position_indices
        )  # [batch_size, num_champions, embed_dim]

        # Sum champion and position embeddings
        champion_inputs = (
            champion_embeds + position_embeds
        )  # [batch_size, num_champions, embed_dim]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            champion_inputs
        )  # [batch_size, num_champions, embed_dim]

        # Pooling: We can use mean pooling over the sequence dimension
        champion_features = transformer_output.mean(dim=1)  # [batch_size, embed_dim]

        embedded_features.append(champion_features)

        # Concatenate all features
        x = torch.cat(embedded_features, dim=1)

        # Pass through fully connected layers
        x = self.fc(x)  # Shared representation

        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs


if __name__ == "__main__":
    # Example usage
    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }
    num_champions = 200

    model = MatchOutcomeModel(
        num_categories=num_categories,
        num_champions=num_champions,
        num_numerical_features=len(NUMERICAL_COLUMNS),
        embed_dim=32,
        dropout=0,
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
