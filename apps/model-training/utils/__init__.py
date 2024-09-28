# utils/__init__.py
import os
import torch


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
DATA_DIR = "./data/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "match_outcome_model.pth")

# Batch sizes
TRAIN_BATCH_SIZE = 512  # Used during training
DATA_EXTRACTION_BATCH_SIZE = 10000  # Used during data extraction from the database
PARQUET_READER_BATCH_SIZE = 10000  # Used when reading data from Parquet files