# serve-model-mock.py
import random
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Literal, Dict, Optional
import uvicorn
import os
from datetime import datetime

# Initialize FastAPI with minimal configuration
app = FastAPI(title="LoL Draft Model Server - Mock Version")

# For authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Mock patch mapping
PATCH_MAPPING = {
    "14.1": 1,
    "14.2": 2,
    "14.3": 3,
    "14.4": 4,
    "14.5": 5,
    "14.6": 6,
    "14.7": 7,
    "14.8": 8,
    "14.9": 9,
    "14.10": 10,
    "14.11": 11,
    "14.12": 12,
    "14.13": 13,
    "14.14": 14,
    "14.15": 15,
    "14.16": 16,
    "14.17": 17,
    "14.18": 18,
    "14.19": 19,
    "14.20": 20,
    "14.21": 21,
    "14.22": 22,
}

# Mock positions
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

# Time buckets for win predictions
TIME_BUCKETS = [
    "win_prediction_0_20",
    "win_prediction_20_25",
    "win_prediction_25_30",
    "win_prediction_30_35",
    "win_prediction_35_40",
    "win_prediction_40_inf",
]


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

        latest_patch = sorted(PATCH_MAPPING.keys())[-1]
        latest_mapped_patch = PATCH_MAPPING[latest_patch]
        if not self.patch:
            # Find the latest patch from patch_mapping
            return latest_mapped_patch

        # Store the mapped patch value
        self._mapped_patch = PATCH_MAPPING.get(self.patch, latest_mapped_patch)
        return self._mapped_patch

    @property
    def elo(self) -> int:
        return self.numerical_elo


class WinratePrediction(BaseModel):
    win_probability: float


@app.post("/predict-batch")
async def predict_batch(inputs: List[APIInput], api_key: str = Depends(verify_api_key)):
    """
    Mock batch prediction endpoint that returns random win probabilities
    """
    # Generate random win probabilities for each input
    win_probabilities = [random.uniform(0.3, 0.7) for _ in inputs]

    # Return predictions
    return [
        WinratePrediction(win_probability=float(prob)) for prob in win_probabilities
    ]


class InDepthPrediction(BaseModel):
    win_probability: float
    gold_diff_15min: List[float]  # [TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY]
    champion_impact: List[float]  # [champ1_impact, champ2_impact, ..., champ10_impact]
    time_bucketed_predictions: Dict[
        str, float
    ]  # Win probabilities for different time buckets


class ModelMetadata(BaseModel):
    patches: List[str]  # List of patches (e.g., ["14.21", "14.22", ...])
    last_modified: str  # ISO format timestamp


@app.post("/predict")
async def predict(input_data: APIInput, api_key: str = Depends(verify_api_key)):
    """Single prediction endpoint"""
    # Reuse batch prediction by calling it with a single item list
    batch_result = await predict_batch([input_data], api_key=api_key)
    return batch_result[0]


@app.post("/predict-in-depth")
async def predict_in_depth(api_input: APIInput, api_key: str = Depends(verify_api_key)):
    """Detailed prediction with champion impact analysis"""
    # Generate a random base win probability
    base_win_prob = random.uniform(0.3, 0.7)

    # Generate random gold differences for each position
    gold_diffs = [random.uniform(-1000, 1000) for _ in POSITIONS]

    # Generate random champion impact values
    # Positive for blue team (0-4), negative for red team (5-9)
    champion_impact = []
    for i in range(10):
        impact = random.uniform(0.01, 0.1)
        if i >= 5:  # Red side champion
            impact = -impact
        champion_impact.append(impact)

    # Generate time-bucketed win predictions
    # These should be somewhat correlated with the base win probability
    time_bucketed_predictions = {}
    for bucket in TIME_BUCKETS:
        # Add some variation to the base win probability
        # but keep it within a reasonable range
        variation = random.uniform(-0.1, 0.1)
        bucket_prob = max(0.1, min(0.9, base_win_prob + variation))
        time_bucketed_predictions[bucket] = bucket_prob

    return InDepthPrediction(
        win_probability=base_win_prob,
        gold_diff_15min=gold_diffs,
        champion_impact=champion_impact,
        time_bucketed_predictions=time_bucketed_predictions,
    )


@app.get("/metadata")
async def get_metadata(api_key: str = Depends(verify_api_key)):
    """Return metadata about the model"""
    # Get current timestamp
    model_timestamp = datetime.now()

    # Extract patches from the patch mapping
    patches = sorted(PATCH_MAPPING.keys())

    return ModelMetadata(patches=patches, last_modified=model_timestamp.isoformat())


if __name__ == "__main__":
    print(f"Starting mock model server on http://0.0.0.0:8000")
    uvicorn.run("serve-model-mock:app", host="0.0.0.0", port=8000, reload=False)
