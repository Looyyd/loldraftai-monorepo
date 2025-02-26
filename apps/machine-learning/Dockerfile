# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory to match local structure
WORKDIR /app/apps/machine-learning

# Copy only the necessary files for inference
COPY requirements-inference.txt requirements.txt
COPY serve-model.py .
COPY utils/ ./utils/

# Create data directory structure to match local
RUN mkdir -p data/models \
    && mkdir -p data/prepared_data \
    && mkdir -p data/raw_azure \
    && mkdir -p data/raw_data

# Copy model files to match local structure
COPY data/models/match_outcome_model.onnx ./data/models/
COPY data/label_encoders.pkl \
     data/model_config.pkl \
     data/numerical_feature_stats.pkl \
     data/task_stats.pkl \
     ./data/
COPY data/prepared_data/patch_mapping.pkl ./data/prepared_data/

# Install dependencies
RUN pip install -r requirements.txt

# Add project root to Python path
ENV PYTHONPATH=/app

# Add API_KEY environment variable with a default value
ENV API_KEY=""

# Expose the port the app runs on
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "serve-model:app", "--host", "0.0.0.0", "--port", "8000"]