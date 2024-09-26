# utils/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.column_definitions import COLUMNS, ColumnType


class MatchOutcomeTransformer(nn.Module):
    def __init__(
        self,
        num_categories,
        num_champions,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(MatchOutcomeTransformer, self).__init__()
        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for col, col_def in COLUMNS.items():
            if col_def.column_type == ColumnType.CATEGORICAL:
                self.embeddings[col] = nn.Embedding(num_categories[col], embed_dim)
                total_embed_dim += embed_dim
            elif col_def.column_type == ColumnType.LIST:
                if col == "champion_ids":
                    self.embeddings[col] = nn.Embedding(num_champions, embed_dim)
                    total_embed_dim += embed_dim * 10  # Assuming 10 champions per match

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Fully connected layers
        self.fc1 = nn.Linear(total_embed_dim, 128)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        embedded_features = []
        for col, embedding_layer in self.embeddings.items():
            if COLUMNS[col].column_type == ColumnType.LIST:
                # For list columns, embed each element and flatten
                embedded = embedding_layer(
                    features[col]
                )  # Shape: [batch_size, seq_length, embed_dim]
                embedded = embedded.view(
                    features[col].size(0), -1
                )  # Flatten to [batch_size, seq_length * embed_dim]
            else:
                embedded = embedding_layer(
                    features[col]
                )  # Shape: [batch_size, embed_dim]
            embedded_features.append(embedded)

        x = torch.cat(embedded_features, dim=1)

        # Pass through fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)

        return x


if __name__ == "__main__":
    # Example usage
    num_categories = {
        col: 10 for col in COLUMNS if COLUMNS[col] == ColumnType.CATEGORICAL
    }
    num_champions = 200

    model = MatchOutcomeTransformer(
        num_categories=num_categories,
        num_champions=num_champions,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )
    print(model)
