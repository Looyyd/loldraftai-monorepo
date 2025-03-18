import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder
from utils.match_prediction import PATCH_MAPPING_PATH, CHAMPION_ID_ENCODER_PATH
from utils.match_prediction.column_definitions import (
    KNOWN_CATEGORICAL_COLUMNS_NAMES,
    COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


class Model(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ):
        super(Model, self).__init__()
        self.embed_dim = embed_dim

        # Load patch mapping and stats
        with open(PATCH_MAPPING_PATH, "rb") as f:
            self.patch_mapping = pickle.load(f)["mapping"]

        with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
            self.champion_id_encoder: LabelEncoder = pickle.load(f)["mapping"]

        self.num_champions = len(self.champion_id_encoder.classes_)
        # Number of unique patches
        self.num_patches = len(self.patch_mapping)

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for col in KNOWN_CATEGORICAL_COLUMNS_NAMES:
            self.embeddings[col] = nn.Embedding(
                len(COLUMNS[col].possible_values), embed_dim
            )

        # Patch embedding (general meta changes)
        self.patch_embedding = nn.Embedding(self.num_patches, embed_dim)

        # Champion+patch embeddings (champion-specific changes)
        self.champion_patch_embedding = nn.Embedding(
            self.num_champions * self.num_patches, embed_dim
        )

        # Total input dimension for MLP
        num_categorical = len(KNOWN_CATEGORICAL_COLUMNS_NAMES)
        num_champions_in_game = len(POSITIONS) * 2  # 10 champions
        total_embed_features = (
            num_categorical + num_champions_in_game + 1
        )  # +1 for patch_embed
        mlp_input_dim = total_embed_features * embed_dim

        print(f"Model dimensions:")
        print(f"- Categorical features: {num_categorical}")
        print(f"- Total embedded features: {total_embed_features}")
        print(f"- Embedding dimension: {embed_dim}")
        print(f"- MLP input dimension: {mlp_input_dim}")

        # MLP
        layers = []
        prev_dim = mlp_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            dropout_rate = dropout if i < len(hidden_dims) - 1 else dropout * 0.5
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Output layers
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

        # Categorical features
        for col in KNOWN_CATEGORICAL_COLUMNS_NAMES:
            embed = self.embeddings[col](features[col])
            embeddings_list.append(embed)

        # Patch embedding (general meta)
        patch_indices = features["patch"]  # (batch_size,)
        patch_embed = self.patch_embedding(patch_indices)  # (batch_size, embed_dim)
        embeddings_list.append(patch_embed)

        # Champion+patch embeddings
        champion_ids = features["champion_ids"]  # (batch_size, 10)
        patch_indices_expanded = patch_indices.unsqueeze(1).expand(
            -1, 10
        )  # (batch_size, 10)
        combined_indices = (
            champion_ids * self.num_patches + patch_indices_expanded
        )  # (batch_size, 10)
        champ_patch_embeds = self.champion_patch_embedding(
            combined_indices
        )  # (batch_size, 10, embed_dim)
        champion_features = champ_patch_embeds.view(
            batch_size, -1
        )  # (batch_size, 10*embed_dim)
        embeddings_list.append(champion_features)

        # Concatenate all features
        combined_features = torch.cat(embeddings_list, dim=1)

        # Pass through MLP
        x = self.mlp(combined_features)

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs
