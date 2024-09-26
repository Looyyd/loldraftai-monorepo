import pickle
from utils import ENCODERS_PATH


def create_inference_features(
    region, tier, division, champion_ids, label_encoders=None
):
    """
    Create features for model inference from input data.

    Args:
    region (str): The region of the match (e.g., 'EUW1', 'NA1')
    tier (str): The average tier of the match (e.g., 'GOLD', 'PLATINUM')
    division (str): The average division of the match (e.g., 'I', 'II')
    champion_ids (list): List of 10 champion IDs in order of positions
    label_encoders (dict, optional): Dictionary of fitted LabelEncoders. If None, load from file.

    Returns:
    dict: A dictionary of features ready for model input
    """
    if label_encoders is None:
        with open(ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)

    try:
        region_encoded = label_encoders["region"].transform([region])[0]
        tier_encoded = label_encoders["averageTier"].transform([tier])[0]
        division_encoded = label_encoders["averageDivision"].transform([division])[0]
    except ValueError as error:
        raise ValueError(f"Error encoding categorical features: {error}")

    if len(champion_ids) != 10:
        raise ValueError("champion_ids must contain exactly 10 champion IDs")

    features = {
        "region": region_encoded,
        "averageTier": tier_encoded,
        "averageDivision": division_encoded,
        "champion_ids": champion_ids,
    }

    return features


# Example usage:
if __name__ == "__main__":
    # This is just for demonstration
    sample_input = {
        "region": "EUW1",
        "tier": "GOLD",
        "division": "II",
        "champion_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Example champion IDs
    }

    features = create_inference_features(**sample_input)
