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
from datetime import datetime
import asyncio

from utils.match_prediction import get_best_device, load_model_state_dict
from utils.match_prediction.model import Model
from utils.match_prediction import (
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    PATCH_MAPPING_PATH,
    CHAMPION_ID_ENCODER_PATH,
    POSITIONS,
    TASK_STATS_PATH,
)
from utils.match_prediction.config import TrainingConfig

device = get_best_device()

# Initialize FastAPI with minimal configuration
app = FastAPI(title="LoL Draft Model Server - GPU Optimized")

# Load resources
print("Loading resources...")

# Load label encoders
with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
    champion_id_encoder = pickle.load(f)["mapping"]

# Load task statistics for denormalization
with open(TASK_STATS_PATH, "rb") as f:
    task_stats = pickle.load(f)

# Get patch mapping
with open(PATCH_MAPPING_PATH, "rb") as f:
    patch_mapping = pickle.load(f)["mapping"]

# Load model config
with open(MODEL_CONFIG_PATH, "rb") as f:
    model_config = pickle.load(f)


# Create model with same architecture
model = Model(
    config=TrainingConfig(),
    hidden_dims=model_config["hidden_dims"],
    dropout=model_config.get(
        "dropout", 0.25
    ),  # Use dropout from config or default to 0.25
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


class InDepthPrediction(BaseModel):
    win_probability: float
    gold_diff_15min: List[float]  # [TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY]
    champion_impact: List[float]  # [champ1_impact, champ2_impact, ..., champ10_impact]


class ModelMetadata(BaseModel):
    patches: List[str]  # List of patches (e.g., ["14.21", "14.22", ...])
    last_modified: str  # ISO format timestamp


def denormalize_value(normalized_value: float, task_name: str) -> float:
    mean = task_stats["means"].get(task_name, 0.0)
    std = task_stats["stds"].get(task_name, 0.0)
    if std == 0:
        return mean
    return (normalized_value * std) + mean


def calculate_gold_differences(model_output: dict) -> List[float]:
    timestamp = "1500000"  # 15 minutes
    gold_diffs = []

    for position in POSITIONS:
        team_100_task = f"team_100_{position}_totalGold_at_{timestamp}"
        team_200_task = f"team_200_{position}_totalGold_at_{timestamp}"

        team_100_gold = denormalize_value(
            float(model_output[team_100_task]), team_100_task
        )
        team_200_gold = denormalize_value(
            float(model_output[team_200_task]), team_200_task
        )

        gold_diffs.append(team_100_gold - team_200_gold)

    return gold_diffs


@app.post("/predict")
async def predict(input_data: APIInput, api_key: str = Depends(verify_api_key)):
    """Single prediction endpoint"""
    # Reuse batch prediction by calling it with a single item list
    batch_result = await predict_batch([input_data], api_key=api_key)
    return batch_result[0]


def monte_carlo_dropout_prediction(model_inputs: dict, n_samples: int = 10000) -> dict:
    """
    Perform Monte Carlo Dropout inference to estimate prediction uncertainty using batched predictions
    """
    # Store original states
    training_states = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Dropout,)):
            training_states[name] = module.training
            module.train()
        else:
            module.eval()

    # Replicate inputs n_samples times
    batched_inputs = {
        k: (
            v.repeat(n_samples, *[1] * (len(v.shape) - 1))
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in model_inputs.items()
    }

    # Single forward pass with the batched inputs
    with torch.no_grad():
        if use_mixed_precision:
            with torch.amp.autocast(device.type):
                outputs = model(batched_inputs)
        else:
            outputs = model(batched_inputs)

        win_pred = torch.sigmoid(outputs["win_prediction"]).cpu().numpy()

    # Calculate statistics
    mean_pred = np.mean(win_pred)

    # Calculate asymmetric 95% confidence intervals
    lower_percentile = np.percentile(win_pred, 2.5)  # 2.5th percentile
    upper_percentile = np.percentile(win_pred, 97.5)  # 97.5th percentile

    # Calculate how far these are from the mean (asymmetric intervals)
    confidence_interval_minus = mean_pred - lower_percentile
    confidence_interval_plus = upper_percentile - mean_pred

    # Restore original states
    for name, module in model.named_modules():
        if name in training_states:
            module.training = training_states[name]
    model.eval()

    return {
        "mean_predictions": mean_pred,
        "confidence_interval_minus": float(confidence_interval_minus),
        "confidence_interval_plus": float(confidence_interval_plus),
        "lower_bound": float(lower_percentile),
        "upper_bound": float(upper_percentile),
    }


@app.post("/predict-in-depth")
async def predict_in_depth(api_input: APIInput, api_key: str = Depends(verify_api_key)):
    """Detailed prediction with champion impact analysis"""
    # Base prediction first
    base_model_inputs = preprocess_batch([api_input])

    # Add Monte Carlo Dropout prediction for testing
    mc_results = monte_carlo_dropout_prediction(base_model_inputs)
    print(f"Monte Carlo Dropout Results:")
    print(f"Mean prediction: {mc_results['mean_predictions']:.3f}")
    print(f"95% CI: [{mc_results['lower_bound']:.3f}, {mc_results['upper_bound']:.3f}]")
    print(
        f"Asymmetric intervals: -{mc_results['confidence_interval_minus']:.3f}/+{mc_results['confidence_interval_plus']:.3f}"
    )

    # Continue with regular prediction...
    with torch.no_grad():
        if use_mixed_precision:
            with torch.amp.autocast(device.type):
                base_outputs = model(base_model_inputs)
        else:
            base_outputs = model(base_model_inputs)

    base_win_pred = base_outputs["win_prediction"]
    base_win_prob = float(torch.sigmoid(base_win_pred).cpu().numpy()[0])

    # Calculate gold differences
    gold_diffs = calculate_gold_differences(base_outputs)

    # Champion impact analysis
    champion_impact = [0.0] * 10
    unknown_champion_id = champion_id_encoder.transform(["UNKNOWN"])[0]

    # Find which champions are known and can be masked
    champion_ids = api_input.champion_ids
    known_champion_indices = []
    for i, champ_id in enumerate(champion_ids):
        if champ_id != "UNKNOWN":
            known_champion_indices.append(i)

    # For each known champion, create a masked version and get prediction
    for idx in known_champion_indices:
        masked_champion_ids = champion_ids.copy()
        masked_champion_ids[idx] = "UNKNOWN"

        masked_input = APIInput(
            champion_ids=masked_champion_ids,
            numerical_elo=api_input.numerical_elo,
            patch=api_input.patch,
            queue_type=api_input.queue_type,
        )

        masked_model_inputs = preprocess_batch([masked_input])
        with torch.no_grad():
            if use_mixed_precision:
                with torch.amp.autocast(device.type):
                    masked_outputs = model(masked_model_inputs)
            else:
                masked_outputs = model(masked_model_inputs)

        masked_win_pred = masked_outputs["win_prediction"]
        masked_win_prob = float(torch.sigmoid(masked_win_pred).cpu().numpy()[0])

        # Calculate impact (positive for blue team, negative for red team)
        impact = base_win_prob - masked_win_prob
        if idx >= 5:  # Red side champion
            impact = -impact

        champion_impact[idx] = impact

    return InDepthPrediction(
        win_probability=base_win_prob,
        gold_diff_15min=gold_diffs,
        champion_impact=champion_impact,
    )


@app.get("/metadata")
async def get_metadata(api_key: str = Depends(verify_api_key)):
    """Return metadata about the model"""
    # Get model file timestamp
    model_timestamp = datetime.fromtimestamp(os.path.getmtime(str(MODEL_PATH)))

    # Extract patches from the patch mapping
    patches = sorted(patch_mapping.keys())

    return ModelMetadata(patches=patches, last_modified=model_timestamp.isoformat())


if __name__ == "__main__":
    print(f"Starting M1-optimized model server on http://0.0.0.0:8000")
    uvicorn.run("serve-model-m1:app", host="0.0.0.0", port=8000, reload=False)
