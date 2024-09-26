# extract-data.py
import os
import pickle
import enum

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sqlalchemy import distinct
from tqdm import tqdm

from utils import TRAIN_DIR, TEST_DIR, ENCODERS_PATH
from utils.database import Match, get_session

# Define positions
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

TEST_SIZE = 0.2  # 20% for testing
BATCH_SIZE = 10000  # Number of records per batch

# Ensure data directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def batch_query(session, batch_size=BATCH_SIZE):
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


def extract_and_save_batches():
    """
    Extract and save batches of data from the database.
    """
    session = get_session()
    label_encoders = {
        "region": LabelEncoder(),
        "averageTier": LabelEncoder(),
        "averageDivision": LabelEncoder(),
    }
    categorical_cols = ["region", "averageTier", "averageDivision"]

    # Collect unique values for label encoding using database queries
    unique_values = {}
    for col in categorical_cols:
        query = session.query(distinct(getattr(Match, col))).filter(
            Match.processed == True,
            Match.processingErrored == False,
            getattr(Match, col) != None,  # Exclude NULL values
        )
        unique_values[col] = [
            value.value if isinstance(value, enum.Enum) else value
            for (value,) in query.all()
            if value
        ]  # Convert Enum to string and remove empty strings if any

    print("Unique values for each column:")
    for col, values in unique_values.items():
        print(f"{col}: {values}")

    # Fit label encoders
    for col in categorical_cols:
        label_encoders[col].fit(unique_values[col])

    # Save label encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(label_encoders, f)
    print(f"Saved label encoders to {ENCODERS_PATH}")

    # Second pass to process and save data batches
    session.close()  # Close and reopen session to reset query
    session = get_session()
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


def extract_features(match, label_encoders):
    """
    Extract features from a match object.
    """
    # Extract basic features
    region = match.region
    tier = match.averageTier
    division = match.averageDivision

    # Extract championIds for each position
    champion_ids = []
    teams = match.teams
    if not teams:
        return None  # Skip if teams data is missing

    # Sort team IDs to ensure consistent order
    for team_id in sorted(teams.keys()):
        participants = teams.get(team_id, {}).get("participants", {})
        for position in POSITIONS:
            participant = participants.get(position, {})
            champion_id = participant.get("championId", 0)
            champion_ids.append(champion_id)

    # Game outcome
    team_100_win = teams.get("100", {}).get("win", False)
    label = 1 if team_100_win else 0

    # Encode categorical features
    try:
        region_encoded = label_encoders["region"].transform(
            [region.value if isinstance(region, enum.Enum) else region]
        )[0]
        tier_encoded = label_encoders["averageTier"].transform(
            [tier.value if isinstance(tier, enum.Enum) else tier]
        )[0]
        division_encoded = label_encoders["averageDivision"].transform(
            [division.value if isinstance(division, enum.Enum) else division]
        )[0]
    except ValueError:
        return None  # Skip if encoding fails

    features = {
        "region": region_encoded,
        "averageTier": tier_encoded,
        "averageDivision": division_encoded,
        "champion_ids": champion_ids,
        "label": label,
    }
    return features


def main():
    extract_and_save_batches()
    print("Data extraction and saving completed.")


if __name__ == "__main__":
    main()
