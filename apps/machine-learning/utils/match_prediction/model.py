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

        # Champion embeddings (general champion characteristics)
        self.champion_embedding = nn.Embedding(
            self.num_champions, config.champion_embed_dim
        )
        mlp_input_dim += config.champion_embed_dim * 10  # 10 champions

        # Champion+patch embeddings (champion-specific patch changes)
        self.champion_patch_embedding = nn.Embedding(
            self.num_champions * self.num_patches, config.champion_patch_embed_dim
        )
        mlp_input_dim += config.champion_patch_embed_dim * 10  # 10 champions

        print(f"MLP input dimension: {mlp_input_dim}")

        # MLP with residual connections
        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_input_dim

        # First layer without residual
        self.mlp_layers.append(
            nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[0], bias=False),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )

        # Middle layers with residual connections
        for i in range(1, len(hidden_dims)):
            current_dim = hidden_dims[i - 1]
            next_dim = hidden_dims[i]

            # Projection layer if dimensions don't match
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, next_dim, bias=False),
                    nn.BatchNorm1d(next_dim),
                    nn.GELU(),
                    nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5),
                )
            )

        # Output layers
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)
            # No need for specific handling for bucketed tasks here,
            # as they are already covered by BINARY_CLASSIFICATION
            elif task_name.startswith("win_prediction_"):
                # Already handled by the BINARY_CLASSIFICATION check above
                pass
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

        # Champion embeddings (general champion characteristics)
        champion_ids = features["champion_ids"]  # (batch_size, 10)
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # (batch_size, 10, config.champion_embed_dim)
        champion_features = champion_embeds.view(
            batch_size, -1
        )  # (batch_size, 10*config.champion_embed_dim)
        embeddings_list.append(champion_features)

        # Champion+patch embeddings (champion-specific patch changes)
        patch_indices_expanded = patch_indices.unsqueeze(1).expand(
            -1, 10
        )  # (batch_size, 10)
        combined_indices = (
            champion_ids * self.num_patches + patch_indices_expanded
        )  # (batch_size, 10)
        champ_patch_embeds = self.champion_patch_embedding(
            combined_indices
        )  # (batch_size, 10, config.champion_patch_embed_dim)
        champ_patch_features = champ_patch_embeds.view(
            batch_size, -1
        )  # (batch_size, 10*config.champion_patch_embed_dim)
        embeddings_list.append(champ_patch_features)

        # Concatenate all features
        x = torch.cat(embeddings_list, dim=1)

        # Pass through MLP with residual connections
        x = self.mlp_layers[0](x)  # First layer without residual

        for i in range(1, len(self.mlp_layers)):
            residual = x
            x = self.mlp_layers[i](x)
            if x.shape == residual.shape:  # Only add residual if shapes match
                x = x + residual

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs
