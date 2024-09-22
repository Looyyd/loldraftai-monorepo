# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchOutcomeTransformer(nn.Module):
    def __init__(
        self,
        num_regions,
        num_tiers,
        num_divisions,
        num_champions,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(MatchOutcomeTransformer, self).__init__()
        # Embeddings
        self.region_emb = nn.Embedding(num_regions, embed_dim)
        self.tier_emb = nn.Embedding(num_tiers, embed_dim)
        self.division_emb = nn.Embedding(num_divisions, embed_dim)
        self.champion_emb = nn.Embedding(num_champions, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Fully connected layers
        # Calculate the input dimension:
        # region_emb + tier_emb + division_emb + champion_emb (flattened)
        # Assume 10 champion_ids (5 per team)
        num_champions_per_match = 10  # Adjust based on your data
        self.fc1 = nn.Linear(embed_dim * (3 + num_champions_per_match), 128)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, region, tier, division, champion_ids):
        # Embedding categorical features
        region = self.region_emb(region)  # (batch_size, embed_dim)
        tier = self.tier_emb(tier)  # (batch_size, embed_dim)
        division = self.division_emb(division)  # (batch_size, embed_dim)

        # Embedding champion IDs
        champion = self.champion_emb(
            champion_ids
        )  # (batch_size, num_champions_per_match, embed_dim)

        # Flatten champion embeddings
        champion = champion.view(
            champion.size(0), -1
        )  # (batch_size, num_champions_per_match * embed_dim)

        # Concatenate all embeddings
        x = torch.cat(
            [region, tier, division, champion], dim=1
        )  # (batch_size, 3*embed_dim + num_champions_per_match*embed_dim)

        # Pass through fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)

        return x


if __name__ == "__main__":
    # Example usage
    num_regions = 17
    num_tiers = 10
    num_divisions = 4
    num_champions = 200
    model = MatchOutcomeTransformer(
        num_regions=num_regions,
        num_tiers=num_tiers,
        num_divisions=num_divisions,
        num_champions=num_champions,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )
    print(model)
