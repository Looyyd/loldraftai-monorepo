# train.py
import os
import pickle
import glob

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import pyarrow.parquet as pq

from utils.match_dataset import MatchDataset
from utils.model import MatchOutcomeTransformer
from utils import get_best_device, TRAIN_DIR, TEST_DIR, ENCODERS_PATH, MODEL_PATH
from utils.column_definitions import COLUMNS, CATEGORICAL_COLUMNS


def collate_fn(batch):
    collated = {col: torch.stack([item[col] for item in batch]) for col in COLUMNS}
    labels = torch.stack([item["label"] for item in batch])
    return collated, labels


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
    train_dataset = MatchDataset(data_dir=TRAIN_DIR, label_encoders_path=ENCODERS_PATH)
    test_dataset = MatchDataset(data_dir=TEST_DIR, label_encoders_path=ENCODERS_PATH)

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

    # Determine the maximum champion ID
    max_champion_id = get_max_champion_id()

    # Determine the number of unique categories from label encoders
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    num_categories = {
        col: len(label_encoders[col].classes_) for col in CATEGORICAL_COLUMNS
    }

    # Initialize the model
    model = MatchOutcomeTransformer(
        num_categories=num_categories,
        num_champions=max_champion_id,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )

    device = get_best_device()
    model.to(device)
    print(f"Using device: {device}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move all features to the device
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)

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
                # Move all features to the device
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)

                outputs = model(features)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
