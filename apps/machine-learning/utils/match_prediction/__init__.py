# utils/match_prediction/__init__.py
import os
import torch

from utils import DATA_DIR

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def get_best_device():
    """
    Get the best device for training the model.(cuda or mps if possible)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


DATABASE_URL = os.getenv("DATABASE_URL")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
RAW_AZURE_DIR = os.path.join(DATA_DIR, "raw_azure")
RAW_PRO_GAMES_DIR = os.path.join(DATA_DIR, "raw_pro_games")
RAW_PRO_GAMES_FILE = os.path.join(RAW_PRO_GAMES_DIR, "pro_games.parquet")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared_data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
# Ensure data directories exist, or create
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(RAW_AZURE_DIR, exist_ok=True)
os.makedirs(RAW_PRO_GAMES_DIR, exist_ok=True)
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.pkl")

MODEL_PATH = os.path.join(MODEL_DIR, "match_outcome_model.pth")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "match_outcome_model.onnx")
MODEL_CONFIG_PATH = os.path.join(DATA_DIR, "model_config.pkl")
TASK_STATS_PATH = os.path.join(DATA_DIR, "task_stats.pkl")
CHAMPION_FEATURES_PATH = os.path.join(DATA_DIR, "champion_features.pkl")
PATCH_MAPPING_PATH = os.path.join(PREPARED_DATA_DIR, "patch_mapping.pkl")
CHAMPION_ID_ENCODER_PATH = os.path.join(DATA_DIR, "champion_id_encoder.pkl")
SAMPLE_COUNTS_PATH = os.path.join(PREPARED_DATA_DIR, "sample_counts.pkl")

DEVICE = get_best_device()

# Batch sizes
if DEVICE.type == "mps":
    TRAIN_BATCH_SIZE = 2048 * 2  # Used during training
else:
    # cuda/runpod config
    TRAIN_BATCH_SIZE = 2048 * 10  # Used during training

# Data batch size could probably be higher
DATA_EXTRACTION_BATCH_SIZE = 512 * 4  # Used during data extraction from the database
PARQUET_READER_BATCH_SIZE = 512 * 4  # Used when reading data from Parquet files


def load_model_state_dict(model: torch.nn.Module, device: torch.device, path: str):
    print(f"Loading pre-trained model from {path}")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    # Remove '_orig_mod.' prefix from state dict keys if present
    fixed_state_dict = {
        k.replace("_orig_mod.", ""): state_dict[k] for k in state_dict.keys()
    }
    model.load_state_dict(fixed_state_dict)
    model.to(device)
    return model
