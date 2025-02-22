# utils/match_prediction/model.py

import torch
import torch.nn as nn
from typing import Dict, List
from utils.match_prediction.column_definitions import (
    NUMERICAL_COLUMNS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType


#######################################
# Model Definition
#######################################
class SimpleMatchModel(nn.Module):
    """
    A modular Transformer-based model that treats each champion
    + each numerical column as its own token in the sequence.

    Architecture:
        1) champion_ids -> Embedding -> shape: [batch_size, #champs, embed_dim]
        2) numerical columns -> separate linear projections -> shape: [batch_size, 1, embed_dim] each
           -> then concatenated into shape: [batch_size, #numerical_cols, embed_dim]
        3) Concatenate all tokens into shape: [batch_size, total_tokens, embed_dim]
           (total_tokens = #champs + #numerical_cols)
        4) Optional token-type embedding (champ vs numeric)
        5) Positional embedding for each token index
        6) Transformer encoder
        7) Pool (e.g. mean) over the sequence dimension
        8) MLP
        9) Task heads
    """

    def __init__(
        self,
        num_champions: int,
        embed_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        num_attention_heads: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        use_token_types: bool = True,
        tasks: Dict[str, Dict] = None,
    ):
        """
        :param num_champions:       Number of unique champion IDs.
        :param embed_dim:           Dimensionality of each embedding token.
        :param hidden_dims:         List of hidden layer sizes for the final MLP.
        :param dropout:             Dropout rate in Transformer & MLP.
        :param num_attention_heads: # of attention heads in Transformer.
        :param num_transformer_layers: # of layers in Transformer.
        :param dim_feedforward:     Feedforward dim in Transformer encoder.
        :param use_token_types:     Whether to include token-type embeddings.
        :param tasks:               Dict of task_name -> { "task_type": TaskType }, used for output heads.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_token_types = use_token_types
        self.tasks = tasks if tasks is not None else TASKS

        ###################################
        # Champion Embeddings
        ###################################
        self.champion_embedding = nn.Embedding(num_champions, embed_dim)

        ###################################
        # Numerical Feature Projections
        #
        # For modularity, we store a separate Linear layer for each numerical column.
        # Each linear takes a single scalar [batch_size, 1] -> [batch_size, embed_dim].
        # Then in forward(), we will .unsqueeze(1) to make it a token.
        ###################################
        self.numerical_embeddings = nn.ModuleDict()
        for col in NUMERICAL_COLUMNS:
            self.numerical_embeddings[col] = nn.Linear(1, embed_dim)

        ###################################
        # Token-Type Embeddings (Optional)
        ###################################
        if self.use_token_types:
            # We'll use:
            #   type 0 => champion tokens
            #   type 1 => numerical tokens
            self.token_type_embedding = nn.Embedding(2, embed_dim)
        else:
            self.token_type_embedding = None

        ###################################
        # Positional Embeddings
        # Total tokens = #champion_ids (e.g., 10) + len(NUMERICAL_COLUMNS)
        ###################################
        self.num_champion_tokens = 10  # or however many champion slots you have
        self.num_numerical_tokens = len(NUMERICAL_COLUMNS)
        self.total_tokens = self.num_champion_tokens + self.num_numerical_tokens
        self.pos_embedding = nn.Parameter(torch.randn(self.total_tokens, embed_dim))

        ###################################
        # Transformer Encoder
        ###################################
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        ###################################
        # MLP (applied after pooling)
        ###################################
        layers = []
        prev_dim = embed_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd, bias=False))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        self.mlp = nn.Sequential(*layers)

        ###################################
        # Output Heads (per task)
        ###################################
        self.output_layers = nn.ModuleDict()
        for task_name, task_info in self.tasks.items():
            task_type = task_info.task_type
            if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)

        print("ContextTransformerModel initialized with:")
        print(f" - #Champion tokens: {self.num_champion_tokens}")
        print(
            f" - #Numerical tokens: {self.num_numerical_tokens} ({NUMERICAL_COLUMNS})"
        )
        print(f" - embed_dim: {embed_dim}, hidden_dims: {hidden_dims}")
        print(f" - num_transformer_layers: {num_transformer_layers}")
        print(f" - num_attention_heads: {num_attention_heads}")
        print(f" - use_token_types: {use_token_types}")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Expects `features` to contain:
          features["champion_ids"] -> LongTensor [batch_size, self.num_champion_tokens]
          features[col] for each col in NUMERICAL_COLUMNS -> FloatTensor [batch_size]
        """
        batch_size = features["champion_ids"].shape[0]

        ###################################
        # 1) Champion Tokens
        ###################################
        # shape: [batch_size, num_champion_tokens, embed_dim]
        champ_embeds = self.champion_embedding(features["champion_ids"])

        ###################################
        # 2) Numerical Tokens
        ###################################
        # For each numerical column, produce a single token.
        numeric_tokens = []
        for col in NUMERICAL_COLUMNS:
            # [batch_size] -> [batch_size, 1]
            vals = features[col].unsqueeze(-1)
            # [batch_size, embed_dim]
            embedded = self.numerical_embeddings[col](vals)
            # Convert to a "token" dimension: [batch_size, 1, embed_dim]
            embedded = embedded.unsqueeze(1)
            numeric_tokens.append(embedded)

        # Concatenate champion tokens + numeric tokens
        # shape => [batch_size, total_tokens, embed_dim]
        # total_tokens = 10 (champs) + len(NUMERICAL_COLUMNS)
        tokens = torch.cat([champ_embeds] + numeric_tokens, dim=1)

        ###################################
        # 3) Token-Type Embeddings (optional)
        ###################################
        if self.use_token_types:
            # 0 for champion tokens, 1 for all numerical tokens
            token_type_ids = (
                torch.tensor(
                    [0] * self.num_champion_tokens + [1] * self.num_numerical_tokens,
                    device=tokens.device,
                )
                .unsqueeze(0)
                .expand(batch_size, self.total_tokens)
            )  # [batch_size, total_tokens]
            type_embeds = self.token_type_embedding(
                token_type_ids
            )  # [batch_size, total_tokens, embed_dim]
            tokens = tokens + type_embeds

        ###################################
        # 4) Positional Embeddings
        #    shape of self.pos_embedding => [total_tokens, embed_dim]
        ###################################
        pos_emb = self.pos_embedding.unsqueeze(0).expand(
            batch_size, self.total_tokens, self.embed_dim
        )
        tokens = tokens + pos_emb

        ###################################
        # 5) Transformer Encoder
        ###################################
        encoded = self.transformer_encoder(
            tokens
        )  # [batch_size, total_tokens, embed_dim]

        ###################################
        # 6) Pooling (Mean pool)
        ###################################
        # shape => [batch_size, embed_dim]
        pooled = encoded.mean(dim=1)

        ###################################
        # 7) MLP
        ###################################
        x = self.mlp(pooled)  # [batch_size, hidden_dims[-1]]

        ###################################
        # 8) Output Heads
        ###################################
        outputs = {}
        for task_name, layer in self.output_layers.items():
            logits = layer(x).squeeze(-1)  # [batch_size]
            outputs[task_name] = logits

        return outputs


#######################################
# Example Usage
#######################################
if __name__ == "__main__":
    # Example: 200 champions
    model = SimpleMatchModel(
        num_champions=200,
        embed_dim=48,
        hidden_dims=[96, 48],
        dropout=0.1,
        num_attention_heads=4,
        num_transformer_layers=2,
        dim_feedforward=128,
        use_token_types=True,
        tasks=TASKS,
    )

    # Create a dummy batch
    batch_size = 8
    feats = {
        "champion_ids": torch.randint(0, 200, (batch_size, 10)),
        "numerical_patch": torch.rand(batch_size),  # e.g. patch number
        "numerical_elo": torch.randint(800, 3000, (batch_size,)).float(),
    }

    out = model(feats)
    for k, v in out.items():
        print(f"{k}: {v.shape}")

    # Print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
