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
from typing import Optional, Tuple
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class Model(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        hidden_dims=None,  # for compatibility with old models
        dropout=None,  # for compatibility with old models
    ):
        super().__init__()

        # Keep your embedding creation code
        # Load champion/patch mappings
        with open(PATCH_MAPPING_PATH, "rb") as f:
            self.patch_mapping = pickle.load(f)["mapping"]
        with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
            self.champion_id_encoder = pickle.load(f)["mapping"]

        self.num_champions = len(self.champion_id_encoder.classes_)
        self.num_patches = len(self.patch_mapping)

        # Use fixed hidden size for all token representations
        self.hidden_size = 128  # Standard size, could be 256 or 512 for more capacity

        # Create embeddings
        self.champion_embedding = nn.Embedding(
            self.num_champions, config.champion_embed_dim
        )
        self.patch_embedding = nn.Embedding(self.num_patches, config.patch_embed_dim)
        self.champion_patch_embedding = nn.Embedding(
            self.num_champions * self.num_patches, config.champion_patch_embed_dim
        )
        self.queue_type_embedding = nn.Embedding(
            len(COLUMNS["queue_type"].possible_values), config.queue_type_embed_dim
        )
        self.elo_embedding = nn.Embedding(
            len(COLUMNS["elo"].possible_values), config.elo_embed_dim
        )

        # Projections to same hidden size
        self.champion_projection = nn.Linear(
            config.champion_embed_dim + config.champion_patch_embed_dim,
            self.hidden_size,
        )
        self.queue_type_projection = nn.Linear(
            config.queue_type_embed_dim, self.hidden_size
        )
        self.patch_projection = nn.Linear(config.patch_embed_dim, self.hidden_size)
        self.elo_projection = nn.Linear(config.elo_embed_dim, self.hidden_size)

        # BERT position embeddings (simpler than RoPE)
        self.seq_length = 13  # 10 champions + queue_type + patch + elo
        self.position_embeddings = nn.Embedding(self.seq_length, self.hidden_size)

        # Layer normalization and dropout (standard in BERT)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # BERT encoder (the transformer stack)
        bert_config = BertConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=4,  # Same as before, adjust as needed
            num_attention_heads=8,
            intermediate_size=self.hidden_size * 4,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
        )
        self.encoder = BertEncoder(bert_config)

        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, config.hidden_dims[-1])

        # Task-specific output heads
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(config.hidden_dims[-1], 1)
            # Skip win_prediction_ handling as before

    def forward(self, features):
        batch_size = features["champion_ids"].size(0)
        device = features["champion_ids"].device

        # Process champion embeddings (first 10 tokens)
        champion_ids = features["champion_ids"]
        patch_indices = features["patch"]
        patch_indices_expanded = patch_indices.unsqueeze(1).expand(-1, 10)
        combined_indices = champion_ids * self.num_patches + patch_indices_expanded

        # Get embeddings
        champion_embeds = self.champion_embedding(champion_ids)
        champion_patch_embeds = self.champion_patch_embedding(combined_indices)

        # Process all sequence tokens
        sequence_tokens = []

        # Champion tokens
        for i in range(10):
            combined_embed = torch.cat(
                [champion_embeds[:, i], champion_patch_embeds[:, i]], dim=-1
            )
            champion_token = self.champion_projection(combined_embed)
            sequence_tokens.append(champion_token.unsqueeze(1))

        # Other tokens
        queue_type_token = self.queue_type_projection(
            self.queue_type_embedding(features["queue_type"])
        ).unsqueeze(1)
        patch_token = self.patch_projection(
            self.patch_embedding(patch_indices)
        ).unsqueeze(1)
        elo_token = self.elo_projection(self.elo_embedding(features["elo"])).unsqueeze(
            1
        )

        sequence_tokens.extend([queue_type_token, patch_token, elo_token])
        sequence = torch.cat(sequence_tokens, dim=1)

        # Add position embeddings
        position_ids = (
            torch.arange(self.seq_length, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeddings = self.position_embeddings(position_ids)

        sequence = sequence + position_embeddings

        # Apply layer norm and dropout (BERT standard)
        sequence = self.layer_norm(sequence)
        sequence = self.dropout(sequence)

        # Create BERT attention mask - all positions attended to
        attention_mask = torch.ones(batch_size, self.seq_length, device=device)

        # Format for BERT
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Pass through BERT encoder
        encoder_outputs = self.encoder(
            sequence, attention_mask=extended_attention_mask, return_dict=True
        )

        # Mean pooling over sequence dimension
        pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)

        # Project to final hidden dimension
        pooled_output = self.output_projection(pooled_output)

        # Task-specific outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(pooled_output).squeeze(-1)

        return outputs
