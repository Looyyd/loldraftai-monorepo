# serve-model.py
import asyncio
from fastapi import FastAPI, HTTPException, Response, Security, Depends
from pydantic import BaseModel
from typing import List, Dict, Literal
import numpy as np
import uvicorn
import pickle
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import onnxruntime as ort
from fastapi.security import APIKeyHeader

from utils.match_prediction.column_definitions import COLUMNS, ColumnType
from utils.match_prediction import (
    CHAMPION_ID_ENCODER_PATH,
    TASK_STATS_PATH,
    MODEL_CONFIG_PATH,
    ONNX_MODEL_PATH,
    POSITIONS,
    PREPARED_DATA_DIR,
    PATCH_MAPPING_PATH,
)


class APIInput(BaseModel):
    champion_ids: List[int | Literal["UNKNOWN"]]
    numerical_elo: int
    patch: str | None = None
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


class ModelInput(BaseModel):
    champion_ids: List[int]
    elo: int
    patch: int
    # For now we default to ranked solo/duo, see column_definitions.py for mapping
    queue_type: int = 0


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
        encoded_champion_ids = champion_id_encoder.transform(
            ["UNKNOWN" if id == -1 else id for id in api_input.champion_ids]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid champion IDs: {str(e)}")

    return ModelInput(
        champion_ids=encoded_champion_ids.tolist(),  # Use encoded champion IDs
        patch=api_input.mapped_patch,
        elo=api_input.elo,
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

# Create the ONNX Runtime session with optimized settings
print(f"Creating ONNX Runtime session from {ONNX_MODEL_PATH}")
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 1  # Optimized for limited CPU resources
session_options.inter_op_num_threads = 1  # Optimized for limited CPU resources

# Create the ONNX session
try:
    onnx_session = ort.InferenceSession(str(ONNX_MODEL_PATH), session_options)
    print("ONNX Runtime session created successfully")

    # Get input and output info from the model
    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]
    input_shapes = {input.name: input.shape for input in onnx_session.get_inputs()}

    print(f"Model inputs: {input_names}")
    print(f"Input shapes: {input_shapes}")
    print(f"Model outputs: {output_names}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    raise

# Create an asyncio.Queue to hold incoming requests
request_queue = asyncio.Queue()

# Initial batch size settings - will be optimized by benchmark
MAX_BATCH_SIZE = 16  # Benchmarked as optimal
BATCH_WAIT_TIME = (
    0.001  # 0.0006686687469482422 benchmarked as optimal, rounded up to 1ms
)


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


def preprocess_input(input_data: ModelInput) -> Dict[str, np.ndarray]:
    """Convert ModelInput to numpy arrays for ONNX Runtime - ensuring correct shapes"""
    processed_input = {}

    # Process categorical features - these should be 1D arrays
    for col, col_def in COLUMNS.items():
        # Process list features - champion_ids is a 2D array
        if col == "champion_ids":
            champion_ids = getattr(input_data, col, [])
            # Make sure it's a 2D array with shape [1, N]
            processed_input[col] = np.array(
                [champion_ids],
                dtype=np.int64,
            )
        # Process known categorical features
        elif col_def.column_type == ColumnType.KNOWN_CATEGORICAL:
            value = getattr(input_data, col)
            processed_input[col] = np.array([value], dtype=np.int64)
        # Handle special columns (patch)
        elif col == "patch":
            processed_input[col] = np.array([input_data.patch], dtype=np.int64)

    return processed_input


# Add near the top of the file
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    expected_key = os.getenv("API_KEY")
    # If API_KEY is empty, skip verification
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key


@app.post("/predict")
async def predict(api_input: APIInput, api_key: str = Depends(verify_api_key)):
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
async def predict_in_depth(api_input: APIInput, api_key: str = Depends(verify_api_key)):
    base_input = api_input_to_model_input(api_input)

    # Get the unknown champion ID
    unknown_champion_id = champion_id_encoder.transform(["UNKNOWN"])[0]

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
    model_timestamp = datetime.fromtimestamp(os.path.getmtime(str(ONNX_MODEL_PATH)))

    # Load patch information
    patch_info = pickle.load(open(PATCH_MAPPING_PATH, "rb"))

    # Convert raw patch numbers to version strings and sort them
    patches = sorted(patch_info["mapping"].keys())

    return ModelMetadata(patches=patches, last_modified=model_timestamp.isoformat())


# Define the background task for ONNX inference
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

        try:
            # Process the batch
            inputs = [r["input"] for r in requests]
            processed_inputs = [preprocess_input(i) for i in inputs]

            # Create batched input for ONNX - handling each input correctly
            batch_size = len(processed_inputs)
            batched_input = {}

            for key in processed_inputs[0].keys():
                # Get the first input to determine shape
                first_input = processed_inputs[0][key]

                if key == "champion_ids":
                    # For champion_ids, we need to preserve the 2D structure (batch_size, num_champions)
                    batched_input[key] = np.vstack(
                        [inp[key] for inp in processed_inputs]
                    )
                elif len(first_input.shape) == 1:
                    # For 1D inputs, we need to ensure we're not adding an extra dimension
                    batched_input[key] = np.concatenate(
                        [inp[key] for inp in processed_inputs]
                    )
                else:
                    # For other inputs, just stack them
                    batched_input[key] = np.vstack(
                        [inp[key] for inp in processed_inputs]
                    )

            # Run ONNX inference
            onnx_outputs = onnx_session.run(None, batched_input)

            # Convert ONNX outputs to dictionary
            output = {name: value for name, value in zip(output_names, onnx_outputs)}

            # Process win probabilities - manually apply sigmoid
            win_probabilities = 1 / (1 + np.exp(-output["win_prediction"]))

            # Return results for each request
            for i, r in enumerate(requests):
                # Extract raw predictions for this request
                # TODO: could maybe return only relevant tasks
                raw_predictions = {
                    task: float(output[task][i]) for task in output.keys()
                }

                r["future"].set_result(
                    {
                        "win_probability": float(win_probabilities[i]),
                        "raw_predictions": raw_predictions,
                    }
                )

        except Exception as e:
            # Handle any errors
            print(f"Error in inference worker: {e}")
            for r in requests:
                if not r["future"].done():
                    r["future"].set_exception(e)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_inference_worker())


@app.post("/predict-batch")
async def predict_batch(inputs: List[APIInput], api_key: str = Depends(verify_api_key)):
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
    uvicorn.run("serve-model:app", host="0.0.0.0", port=8000)
