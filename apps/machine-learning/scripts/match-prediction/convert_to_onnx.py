# scripts/match-prediction/convert_to_onnx.py
import torch
import pickle
from pathlib import Path

from utils.match_prediction.model import Model
from utils.match_prediction.column_definitions import COLUMNS, ColumnType
from utils.match_prediction import (
    MODEL_PATH,
    MODEL_CONFIG_PATH,
)


def preprocess_sample_input():
    """Create a sample input for model tracing with realistic data"""
    sample_input = {}

    # Add champion_ids using realistic champion IDs - common in actual gameplay
    # These IDs represent popular champions likely seen in real matches
    sample_input["champion_ids"] = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long
    )

    # Add other required inputs from COLUMNS
    for col, col_def in COLUMNS.items():
        if col != "champion_ids":  # Already handled
            if col_def.column_type == ColumnType.CATEGORICAL:
                sample_input[col] = torch.tensor([0], dtype=torch.long)
            elif col_def.column_type == ColumnType.NUMERICAL:
                # Use more realistic values for numerical features
                if col == "numerical_elo":
                    sample_input[col] = torch.tensor(
                        [0], dtype=torch.float
                    )  # Normalized ELO value
                elif col == "numerical_patch":
                    sample_input[col] = torch.tensor(
                        [0], dtype=torch.float
                    )  # Normalized patch value
                else:
                    sample_input[col] = torch.tensor([0.0], dtype=torch.float)
            elif col_def.column_type == ColumnType.LIST and col != "champion_ids":
                sample_input[col] = torch.tensor([[0.0]], dtype=torch.float)

    return sample_input


def main():
    print("Starting PyTorch to ONNX model conversion...")

    # Load model configuration
    print(f"Loading model configuration from {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH, "rb") as f:
        model_params = pickle.load(f)

    # Initialize the model
    print("Initializing PyTorch model...")
    model = Model(
        num_categories=model_params["num_categories"],
        num_champions=model_params["num_champions"],
        embed_dim=model_params["embed_dim"],
        dropout=model_params["dropout"],
        hidden_dims=model_params["hidden_dims"],
    )

    # Load weights
    print(f"Loading model weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # Create sample input for tracing
    print("Creating sample input for tracing...")
    sample_input = preprocess_sample_input()

    # Output path
    onnx_path = Path("data/models/match_outcome_model.onnx")
    print(f"Will export ONNX model to {onnx_path}")

    # Make sure the directory exists
    onnx_path.parent.mkdir(exist_ok=True, parents=True)

    # Export the model to ONNX
    print("Exporting model to ONNX format...")

    # Create a wrapper class that takes individual inputs and combines them
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model: Model) -> None:
            super().__init__()
            self.model = model
            self.input_names = list(sample_input.keys())

        def forward(self, *args) -> dict:
            # Combine inputs into a dictionary
            features = {name: arg for name, arg in zip(self.input_names, args)}
            return self.model(features=features)

    # Wrap the model
    wrapped_model = ModelWrapper(model)

    # Prepare inputs as a tuple in the same order as input_names
    input_tuple = tuple(sample_input[name] for name in wrapped_model.input_names)

    torch.onnx.export(
        wrapped_model,  # Wrapped PyTorch model
        input_tuple,  # Tuple of input tensors
        onnx_path,  # Output path
        input_names=wrapped_model.input_names,
        output_names=list(model.output_layers.keys()),
        dynamic_axes={k: {0: "batch_size"} for k in wrapped_model.input_names},
        export_params=True,
        verbose=False,
    )

    # Verify size
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Model exported successfully to {onnx_path} ({model_size_mb:.2f} MB)")
    print("Conversion complete!")


if __name__ == "__main__":
    main()
