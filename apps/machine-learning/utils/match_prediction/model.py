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
from utils.match_prediction.task_definitions import TASKS, TaskType, CONDITIONAL_TASKS
from utils.match_prediction.config import TrainingConfig


class Model(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ):
        super(Model, self).__init__()

        # Load patch mapping and stats
        with open(PATCH_MAPPING_PATH, "rb") as f:
            self.patch_mapping = pickle.load(f)["mapping"]

        with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
            self.champion_id_encoder: LabelEncoder = pickle.load(f)["mapping"]

        self.num_champions = len(self.champion_id_encoder.classes_)
        # Number of unique patches
        self.num_patches = len(self.patch_mapping)

        # Embeddings for categorical features
        mlp_input_dim = 0
        self.embeddings = nn.ModuleDict()
        for col in KNOWN_CATEGORICAL_COLUMNS_NAMES:
            if col == "queue_type":
                embed_dim = config.queue_type_embed_dim
            elif col == "elo":
                embed_dim = config.elo_embed_dim
            else:
                raise ValueError(f"Unhandled categorical column: {col}")
            mlp_input_dim += embed_dim
            self.embeddings[col] = nn.Embedding(
                len(COLUMNS[col].possible_values), embed_dim
            )

        # Patch embedding (general meta changes)
        self.patch_embedding = nn.Embedding(self.num_patches, config.patch_embed_dim)
        mlp_input_dim += config.patch_embed_dim
        # Champion+patch embeddings (champion-specific changes)
        self.champion_patch_embedding = nn.Embedding(
            self.num_champions * self.num_patches, config.champion_embed_dim
        )
        mlp_input_dim += config.champion_embed_dim * 10  # 10 champions

        print(f"MLP input dimension: {mlp_input_dim}")

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
        for task_name, task_def in [*TASKS.items(), *CONDITIONAL_TASKS.items()]:
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)
            else:
                raise ValueError(f"Unknown task type: {task_def.task_type}")

    def forward(self, features):
        batch_size = features["champion_ids"].size(0)
        embeddings_list = []

        # Categorical features
        for col in KNOWN_CATEGORICAL_COLUMNS_NAMES:
            embed = self.embeddings[col](features[col])
            embeddings_list.append(embed)

        # Patch embedding (general meta)
        patch_indices = features["patch"]  # (batch_size,)
        patch_embed = self.patch_embedding(
            patch_indices
        )  # (batch_size, config.patch_embed_dim)
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
        )  # (batch_size, 10, config.champion_embed_dim)
        champion_features = champ_patch_embeds.view(
            batch_size, -1
        )  # (batch_size, 10*config.champion_embed_dim)
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
