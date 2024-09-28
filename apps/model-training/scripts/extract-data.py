# scripts/extract-data.py
import os
import pickle
import enum
import shutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sqlalchemy import distinct
from tqdm import tqdm
import numpy as np
from utils import TRAIN_DIR, TEST_DIR, ENCODERS_PATH, DATA_EXTRACTION_BATCH_SIZE
from utils.database import Match, get_session
from utils.column_definitions import (
    COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    extract_raw_features,
    ColumnType,
)


# Define positions
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

TEST_SIZE = 0.2  # 20% for testing

# Ensure data directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def batch_query(session, batch_size=DATA_EXTRACTION_BATCH_SIZE):
    """
    Generator that yields batches of matches from the database.
    """
    last_id = None
    while True:
        query = session.query(Match).filter(
            Match.processed == True, Match.processingErrored == False
        )
        if last_id:
            query = query.filter(Match.id > last_id)
        query = query.order_by(Match.id).limit(batch_size)
        matches = query.all()
        if not matches:
            break
        last_id = matches[-1].id
        yield matches


def create_label_encoders(session):
    """
    Create and fit label encoders for all categorical columns.
    """
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        print(f"Creating label encoder for {col}")
        query = session.query(distinct(getattr(Match, col))).filter(
            Match.processed == True,
            Match.processingErrored == False,
            getattr(Match, col) != None,  # Exclude NULL values
        )
        unique_values = [
            value.value if isinstance(value, enum.Enum) else value
            for (value,) in query.all()
            if value is not None
        ]
        unique_values = sorted(
            set(unique_values)
        )  # Ensure uniqueness and consistent ordering

        # Add a special 'UNKNOWN' value to handle unseen categories
        unique_values.append("UNKNOWN")

        print(f"Unique values for {col}: {unique_values}")

        encoder = LabelEncoder()
        encoder.fit(unique_values)
        label_encoders[col] = encoder

    return label_encoders


def extract_and_save_batches():
    """
    Extract and save batches of data from the database.
    """
    session = get_session()

    # Clean up previous data
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Create and fit label encoders
    label_encoders = create_label_encoders(session)

    # Save label encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"Saved label encoders to {ENCODERS_PATH}")

    # Process and save data batches
    batch_num = 0
    for matches in tqdm(batch_query(session), desc="Processing and saving batches"):
        data = []
        for match in matches:
            # Extract features
            features = extract_features(match, label_encoders)
            if features:
                data.append(features)
        if data:
            df = pd.DataFrame(data)
            # Split into train/test
            df_train, df_test = train_test_split(
                df, test_size=TEST_SIZE, random_state=42
            )
            # Save to Parquet
            train_file = os.path.join(TRAIN_DIR, f"train_batch_{batch_num}.parquet")
            test_file = os.path.join(TEST_DIR, f"test_batch_{batch_num}.parquet")
            df_train.to_parquet(train_file, index=False)
            df_test.to_parquet(test_file, index=False)
            batch_num += 1

    session.close()


def extract_features(match: Match, label_encoders: dict):
    features = extract_raw_features(match)

    if len(features["champion_ids"]) != 10:
        return None  # Skip this sample if it doesn't meet expectations

    # Apply label encoding for categorical features
    for col, def_ in COLUMNS.items():
        if def_.column_type == ColumnType.CATEGORICAL:
            features[col] = label_encoders[col].transform([features[col]])[0]

    for col in NUMERICAL_COLUMNS:
        if features[col] == -1:
            return None  # Skip samples with missing numerical data

    # Add the label
    features["label"] = 1 if match.teams.get("100", {}).get("win", False) else 0

    return features


def main():
    extract_and_save_batches()
    print("Data extraction and saving completed.")


if __name__ == "__main__":
    main()
