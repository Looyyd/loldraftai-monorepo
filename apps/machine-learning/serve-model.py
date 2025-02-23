import asyncio
from fastapi import FastAPI, HTTPException, Response, Security, Depends
from pydantic import BaseModel
from typing import List, Dict, Literal
import torch
import uvicorn
import pickle
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from fastapi.security import APIKeyHeader

from utils.match_prediction.model import Model
from utils.match_prediction.column_definitions import COLUMNS, ColumnType
from utils.match_prediction import (
    MODEL_PATH,
    ENCODERS_PATH,
    NUMERICAL_STATS_PATH,
    TASK_STATS_PATH,
    MODEL_CONFIG_PATH,
    POSITIONS,
    PREPARED_DATA_DIR,
    get_best_device,
)


class APIInput(BaseModel):
    champion_ids: List[int | Literal["UNKNOWN"]]
    numerical_elo: int
    patch: str | None = None
    _mapped_patch: int | None = None

    @property
    def numerical_patch(self) -> int:
        """Convert patch string (e.g., '14.22') to numerical format"""
        if self._mapped_patch is not None:
            return self._mapped_patch

        if not self.patch:
            # Find the latest patch from patch_mapping
            latest_patch = max(patch_mapping.keys())
            return int(latest_patch)

        major, minor = self.patch.split(".")
        raw_patch = int(major) * 50 + int(minor)
        # Store the mapped patch value
        self._mapped_patch = patch_mapping.get(float(raw_patch), 0)
        return self._mapped_patch

    # TODO: could try adding validator


class ModelInput(APIInput):
    champion_ids: List[int]
    pass


class InDepthPrediction(BaseModel):
    win_probability: float
    gold_diff_15min: List[float]  # [TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY]
    champion_impact: List[float]  # [champ1_impact, champ2_impact, ..., champ10_impact]


class WinratePrediction(BaseModel):
    win_probability: float


class ModelMetadata(BaseModel):
    patches: List[str]  # List of patches (e.g., ["14.21", "14.22", ...])
    last_modified: str  # ISO format timestamp


def api_input_to_model_input(api_input: APIInput) -> ModelInput:
    try:
        # Convert -1 to "UNKNOWN" and encode all champion IDs in one go
        encoded_champion_ids = label_encoders["champion_ids"].transform(
            ["UNKNOWN" if id == -1 else id for id in api_input.champion_ids]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid champion IDs")

    # Create a dictionary from the API input, excluding 'champion_ids'
    input_dict = api_input.model_dump(exclude={"champion_ids"})

    return ModelInput(
        **input_dict,
        champion_ids=encoded_champion_ids.tolist(),  # Use encoded champion IDs
    )


app = FastAPI()

# Add CORS middleware configuration right after creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load label encoders
with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

# Load numerical stats for normalization
with open(NUMERICAL_STATS_PATH, "rb") as f:
    numerical_stats = pickle.load(f)

# Load task statistics for denormalization
with open(TASK_STATS_PATH, "rb") as f:
    task_stats = pickle.load(f)

with open(MODEL_CONFIG_PATH, "rb") as f:
    model_params = pickle.load(f)

# Load the model
device = get_best_device()
device = "cpu"
print(f"Using device: {device}")

model = Model(
    num_categories=model_params["num_categories"],
    num_champions=model_params["num_champions"],
    embed_dim=model_params["embed_dim"],
    dropout=model_params["dropout"],
    hidden_dims=model_params["hidden_dims"],
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()


with open(Path(PREPARED_DATA_DIR) / "patch_mapping.pkl", "rb") as f:
    patch_mapping = pickle.load(f)["mapping"]


# Create an asyncio.Queue to hold incoming requests
request_queue = asyncio.Queue()

# Max batch size and batch wait time
MAX_BATCH_SIZE = 32
BATCH_WAIT_TIME = 0.05  # in seconds


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


def preprocess_input(input_data: ModelInput) -> Dict[str, torch.Tensor]:
    processed_input = {}
    for col, col_def in COLUMNS.items():
        if col_def.column_type == ColumnType.CATEGORICAL:
            value = getattr(input_data, col)
            if value not in label_encoders[col].classes_:
                value = "UNKNOWN"
            processed_input[col] = torch.tensor(
                [label_encoders[col].transform([value])[0]],
                dtype=torch.long,
                device=device,
            )
        elif col_def.column_type == ColumnType.NUMERICAL:
            value = getattr(input_data, col)
            mean = numerical_stats["means"].get(col, 0.0)
            std = numerical_stats["stds"].get(col, 1.0)
            if std == 0:
                normalized_value = 0.0  # Avoid division by zero
            else:
                normalized_value = (value - mean) / std
            processed_input[col] = torch.tensor(
                [normalized_value], dtype=torch.float, device=device
            )
        elif col_def.column_type == ColumnType.LIST:
            if col == "champion_ids":
                processed_input[col] = torch.tensor(
                    [getattr(input_data, col)], dtype=torch.long, device=device
                )
            else:
                processed_input[col] = torch.tensor(
                    [getattr(input_data, col)], dtype=torch.float, device=device
                )

    return processed_input


# Add near the top of the file
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key


@app.post("/predict")
async def predict(
    api_input: APIInput, api_key: str = Depends(verify_api_key)  # Add this parameter
):
    model_input = api_input_to_model_input(api_input)

    # Create a future to hold the response
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    # Put the request into the queue
    await request_queue.put({"input": model_input, "future": future})
    # Wait for the result
    result = await future
    return WinratePrediction(win_probability=result["win_probability"])


@app.post("/predict-in-depth")
async def predict_in_depth(
    api_input: APIInput, api_key: str = Depends(verify_api_key)  # Add this parameter
):
    base_input = api_input_to_model_input(api_input)

    # Get the unknown champion ID
    unknown_champion_id = label_encoders["champion_ids"].transform(["UNKNOWN"])[0]

    # Create masked versions only for known champions
    masked_inputs = []
    known_champion_indices = []
    for i, champ_id in enumerate(base_input.champion_ids):
        if champ_id != unknown_champion_id:
            masked_champion_ids = base_input.champion_ids.copy()
            masked_champion_ids[i] = unknown_champion_id
            masked_inputs.append(
                ModelInput(
                    **{
                        k: v
                        for k, v in base_input.model_dump().items()
                        if k != "champion_ids"
                    },
                    champion_ids=masked_champion_ids,
                )
            )
            known_champion_indices.append(i)

    # Create futures for predictions
    loop = asyncio.get_event_loop()
    futures = []

    # Queue base prediction
    base_future = loop.create_future()
    await request_queue.put({"input": base_input, "future": base_future})
    futures.append(base_future)

    # Queue masked predictions only for known champions
    for masked_input in masked_inputs:
        masked_future = loop.create_future()
        await request_queue.put({"input": masked_input, "future": masked_future})
        futures.append(masked_future)

    # Wait for all results
    results = await asyncio.gather(*futures)

    # Extract base prediction and masked predictions
    base_result = results[0]
    masked_results = results[1:]

    # Initialize champion impact array with zeros
    champion_impact = [0.0] * 10

    # Calculate champion impact only for known champions
    base_winrate = base_result["win_probability"]
    for idx, masked_result in zip(known_champion_indices, masked_results):
        impact = base_winrate - masked_result["win_probability"]
        if idx >= 5:  # Red side champions
            impact = -impact
        champion_impact[idx] = impact

    return InDepthPrediction(
        win_probability=base_winrate,
        gold_diff_15min=calculate_gold_differences(base_result["raw_predictions"]),
        champion_impact=champion_impact,
    )


@app.get("/metadata")
async def get_metadata(api_key: str = Depends(verify_api_key)):
    # Get model file timestamp
    model_timestamp = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))

    # Load patch information
    patch_info = pickle.load(open(Path(PREPARED_DATA_DIR) / "patch_mapping.pkl", "rb"))

    # Convert raw patch numbers to version strings and sort them
    patches = sorted(
        f"{patch // 50}.{patch % 50:02d}" for patch in patch_info["mapping"].keys()
    )

    return ModelMetadata(patches=patches, last_modified=model_timestamp.isoformat())


# Define the background task
async def model_inference_worker():
    while True:
        requests = []
        # Wait until there is at least one item in the queue
        request = await request_queue.get()
        requests.append(request)
        start_time = asyncio.get_event_loop().time()
        # Wait for more items in the queue up to MAX_BATCH_SIZE or BATCH_WAIT_TIME
        while len(requests) < MAX_BATCH_SIZE:
            try:
                # Wait for the remaining time
                timeout = BATCH_WAIT_TIME - (
                    asyncio.get_event_loop().time() - start_time
                )
                if timeout <= 0:
                    break
                request = await asyncio.wait_for(request_queue.get(), timeout)
                requests.append(request)
            except asyncio.TimeoutError:
                break
        # Process the batch
        inputs = [r["input"] for r in requests]
        processed_inputs = [preprocess_input(i) for i in inputs]
        batched_input = {
            key: torch.cat([inp[key] for inp in processed_inputs], dim=0)
            for key in processed_inputs[0].keys()
        }

        with torch.inference_mode():
            output = model(batched_input)

        win_probabilities = (
            torch.sigmoid(output["win_prediction"]).cpu().numpy().tolist()
        )
        # Return all raw predictions for each request
        for i, r in enumerate(requests):
            # TODO: could maybe return only relevant tasks
            raw_predictions = {
                task: output[task][i].cpu().item() for task in output.keys()
            }
            r["future"].set_result(
                {
                    "win_probability": win_probabilities[i],
                    "raw_predictions": raw_predictions,
                }
            )


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_inference_worker())


@app.post("/predict-batch")
async def predict_batch(
    inputs: List[APIInput], api_key: str = Depends(verify_api_key)  # Add this parameter
):
    # Create futures for all predictions
    loop = asyncio.get_event_loop()
    futures = []

    for api_input in inputs:
        model_input = api_input_to_model_input(api_input)
        future = loop.create_future()
        await request_queue.put({"input": model_input, "future": future})
        futures.append(future)

    # Wait for all results
    results = await asyncio.gather(*futures)
    return [
        WinratePrediction(win_probability=result["win_probability"])
        for result in results
    ]


# For running with uvicorn
if __name__ == "__main__":
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000)
