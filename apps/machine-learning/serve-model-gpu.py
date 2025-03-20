# serve-model-m1.py
import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Literal, Dict, Optional
import uvicorn
import pickle
import os
from pathlib import Path
import numpy as np

from utils.match_prediction import get_best_device, load_model_state_dict
from utils.match_prediction.model import Model
from utils.match_prediction import (
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    PATCH_MAPPING_PATH,
    CHAMPION_ID_ENCODER_PATH,
)

device = get_best_device()

# Initialize FastAPI with minimal configuration
app = FastAPI(title="LoL Draft Model Server - GPU Optimized")

# Load resources
print("Loading resources...")

# Load label encoders
with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
    champion_id_encoder = pickle.load(f)["mapping"]


# Get patch mapping
with open(PATCH_MAPPING_PATH, "rb") as f:
    patch_mapping = pickle.load(f)["mapping"]

# Load model config
with open(MODEL_CONFIG_PATH, "rb") as f:
    model_config = pickle.load(f)


# Create model with same architecture
model = Model(
    embed_dim=model_config["embed_dim"],
    hidden_dims=model_config["hidden_dims"],
    dropout=0.0,  # No dropout needed for inference
)

model = load_model_state_dict(model, device, path=MODEL_PATH)
model.eval()
print("Model loaded successfully")

# Enable mixed precision when possible (faster on GPUs)
use_mixed_precision = False
if device.type in ["cuda"]:
    print(f"Enabling mixed precision for faster inference on {device.type}")
    use_mixed_precision = True

# For authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    expected_key = os.getenv("API_KEY")
    # If API_KEY is empty, skip verification
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class APIInput(BaseModel):
    champion_ids: List[int | Literal["UNKNOWN"]]
    numerical_elo: int
    patch: str | None = None
    queue_type: int = 0
    _mapped_patch: int | None = None

    @property
    def mapped_patch(self) -> int:
        """Convert patch string (e.g., '14.22') to numerical format"""
        if self._mapped_patch is not None:
            return self._mapped_patch

        latest_patch = sorted(patch_mapping.keys())[-1]
        latest_mapped_patch = patch_mapping[latest_patch]
        if not self.patch:
            # Find the latest patch from patch_mapping
            return latest_mapped_patch

        # Store the mapped patch value
        self._mapped_patch = patch_mapping.get(self.patch, latest_mapped_patch)
        return self._mapped_patch

    # TODO: could be removed(but also in frontend), artifact of old model architecture
    @property
    def elo(self) -> int:
        return self.numerical_elo


class WinratePrediction(BaseModel):
    win_probability: float


def preprocess_batch(batch_inputs: List[APIInput]):
    """
    Efficiently preprocess a batch of inputs for the model
    """
    # Process champion IDs
    all_champion_ids = []
    for input_data in batch_inputs:
        # Handle "UNKNOWN" champions properly
        ids = [id if id != "UNKNOWN" else "UNKNOWN" for id in input_data.champion_ids]
        all_champion_ids.append(ids)

    # Convert to encoded values
    encoded_champions = []
    for champion_ids in all_champion_ids:
        # Transform directly with label encoder handling "UNKNOWN"
        encoded = champion_id_encoder.transform(champion_ids)
        encoded_champions.append(encoded)

    # Convert list to numpy array first, then to tensor
    encoded_champions = np.array(encoded_champions)
    champion_ids_tensor = torch.tensor(
        encoded_champions, dtype=torch.long, device=device
    )

    elos = [input_data.elo for input_data in batch_inputs]

    elo_tensor = torch.tensor(elos, dtype=torch.long, device=device)

    patches = [input_data.mapped_patch for input_data in batch_inputs]

    patch_tensor = torch.tensor(patches, dtype=torch.long, device=device)

    # Process queueId
    queues = [input_data.queue_type for input_data in batch_inputs]

    queue_tensor = torch.tensor(queues, dtype=torch.long, device=device)

    # Return as dictionary for model input
    return {
        "champion_ids": champion_ids_tensor,
        "elo": elo_tensor,
        "patch": patch_tensor,
        "queue_type": queue_tensor,
    }


@app.post("/predict-batch")
async def predict_batch(inputs: List[APIInput], api_key: str = Depends(verify_api_key)):
    """
    Optimized batch prediction endpoint
    """
    model_inputs = preprocess_batch(inputs)
    with torch.no_grad():
        # Use mixed precision if enabled and available
        if use_mixed_precision:
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(model_inputs)
            elif device.type == "mps":
                outputs = model(model_inputs)
        else:
            outputs = model(model_inputs)

        # Get win predictions
        win_predictions = outputs["win_prediction"]
        win_probabilities = torch.sigmoid(win_predictions).cpu().numpy()

    # Return predictions
    return [
        WinratePrediction(win_probability=float(prob)) for prob in win_probabilities
    ]


if __name__ == "__main__":
    print(f"Starting M1-optimized model server on http://0.0.0.0:8000")
    uvicorn.run("serve-model-m1:app", host="0.0.0.0", port=8000, reload=False)
