# scripts/match-prediction/generate_playrates.py
# Generates champion play rates for the latest 10 patches
# Playrates are used by the frontend to :
# - Determine in what role to place a champion automatically
# - Show a warning for rare champion/role combinations
import os
import glob
import json
import pandas as pd
from typing import Dict, DefaultDict
from collections import defaultdict
from tqdm import tqdm
from utils.match_prediction import RAW_AZURE_DIR
from utils import champion_play_rates_path
from utils.match_prediction.column_definitions import (
    COLUMNS,
    get_champion_ids,
    POSITIONS,
)


def calculate_play_rates(input_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate champion play rates per patch and role.

    Args:
        input_dir: Directory containing raw parquet files

    Returns:
        Dict with structure: {patch: {champion_id: {role: play_rate}}}
    """
    # Initialize counters
    games_per_patch: DefaultDict[str, int] = defaultdict(int)
    champ_role_counts: DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    # Process all parquet files
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))

    for file_path in tqdm(input_files, desc="Processing files"):
        df = pd.read_parquet(file_path)

        # Create patch version string
        df["patch"] = (
            df["gameVersionMajorPatch"].astype(str)
            + "."
            + df["gameVersionMinorPatch"].astype(str).str.zfill(2)
        )

        # Count total games per patch
        patch_counts = df["patch"].value_counts()
        for patch, count in patch_counts.items():
            games_per_patch[patch] += count

        # Get champion IDs using the same function from column_definitions
        champion_ids = get_champion_ids(df)

        # Process champion appearances
        for idx, row in df.iterrows():
            patch = row["patch"]
            champs = champion_ids.iloc[idx]  # Get champion IDs for this game

            # Process blue team (first 5 champions)
            for pos, role in enumerate(POSITIONS):
                champ_id = str(int(champs[pos]))
                champ_role_counts[patch][champ_id][role] += 1

            # Process red team (last 5 champions)
            for pos, role in enumerate(POSITIONS):
                champ_id = str(int(champs[pos + 5]))
                champ_role_counts[patch][champ_id][role] += 1

    # Calculate play rates
    play_rates: Dict[str, Dict[str, Dict[str, float]]] = {}

    for patch in champ_role_counts:
        play_rates[patch] = {}
        total_games = games_per_patch[patch]

        for champ_id in champ_role_counts[patch]:
            play_rates[patch][champ_id] = {}

            for role in POSITIONS:
                count = champ_role_counts[patch][champ_id][role]
                play_rate = (count / total_games) * 100  # Convert to percentage
                play_rates[patch][champ_id][role] = round(play_rate, 3)

    return play_rates


def main():
    # Calculate play rates
    play_rates = calculate_play_rates(RAW_AZURE_DIR)

    # Sort patches by version (newest first)
    sorted_play_rates = dict(
        sorted(
            play_rates.items(),
            key=lambda x: tuple(map(float, x[0].split("."))),
            reverse=True,
        )
    )

    # Limit to only the latest 10 patches
    latest_5_patches = dict(list(sorted_play_rates.items())[:5])

    # Save to JSON
    os.makedirs(os.path.dirname(champion_play_rates_path), exist_ok=True)

    with open(champion_play_rates_path, "w") as f:
        json.dump(latest_5_patches, f, indent=2)

    print(f"Play rates saved to {champion_play_rates_path}")
    print(
        f"Processed {len(play_rates)} patches, saved {len(latest_5_patches)} most recent patches"
    )


if __name__ == "__main__":
    main()
