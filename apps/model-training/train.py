# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from match_dataset import MatchDataset
from model import MatchOutcomeTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
import pyarrow.parquet as pq
import pickle
import glob
import pandas as pd

DATA_DIR = "./data/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABEL_ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.pkl")


def collate_fn(batch):
    """
    Custom collate function to handle batching.
    """
    regions = torch.stack([item["region"] for item in batch])
    average_tiers = torch.stack([item["averageTier"] for item in batch])
    average_divisions = torch.stack([item["averageDivision"] for item in batch])
    champion_ids = torch.stack([item["champion_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {
        "region": regions,
        "averageTier": average_tiers,
        "averageDivision": average_divisions,
        "champion_ids": champion_ids,
    }, labels


def get_max_champion_id():
    # Function to get the maximum champion ID from the data
    max_id = 0
    for dir_path in [TRAIN_DIR, TEST_DIR]:
        data_files = glob.glob(os.path.join(dir_path, "*.parquet"))
        for file_path in data_files:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=10000):
                df_chunk = batch.to_pandas()
                max_id_in_chunk = df_chunk["champion_ids"].apply(max).max()
                if max_id_in_chunk > max_id:
                    max_id = max_id_in_chunk
    return max_id + 1  # +1 for padding if needed


def train_model():
    # Initialize the datasets
    train_dataset = MatchDataset(
        data_dir=TRAIN_DIR, label_encoders_path=LABEL_ENCODERS_PATH
    )
    test_dataset = MatchDataset(
        data_dir=TEST_DIR, label_encoders_path=LABEL_ENCODERS_PATH
    )

    # Initialize the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=4,  # Adjust based on your CPU
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, num_workers=4, collate_fn=collate_fn
    )

    # Determine the number of unique categories from label encoders
    with open(LABEL_ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    num_regions = len(label_encoders["region"].classes_)
    num_tiers = len(label_encoders["averageTier"].classes_)
    num_divisions = len(label_encoders["averageDivision"].classes_)

    # Determine the maximum champion ID
    max_champion_id = get_max_champion_id()

    # Initialize the model
    model = MatchOutcomeTransformer(
        num_regions=num_regions,
        num_tiers=num_tiers,
        num_divisions=num_divisions,
        num_champions=max_champion_id,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            region = features["region"].to(device)
            average_tier = features["averageTier"].to(device)
            average_division = features["averageDivision"].to(device)
            champion_ids = features["champion_ids"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(region, average_tier, average_division, champion_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * region.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}], Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in test_loader:
                region = features["region"].to(device)
                average_tier = features["averageTier"].to(device)
                average_division = features["averageDivision"].to(device)
                champion_ids = features["champion_ids"].to(device)
                labels = labels.to(device)

                outputs = model(region, average_tier, average_division, champion_ids)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Save the model
    MODEL_PATH = os.path.join(DATA_DIR, "match_outcome_model.pth")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
