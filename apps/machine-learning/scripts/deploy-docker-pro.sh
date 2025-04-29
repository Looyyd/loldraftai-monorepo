#!/bin/bash

# 1. Login to Azure Container Registry (if not already logged in)
az acr login --name leaguedraftv2registrypro

# Build pro onnx model
python scripts/match-prediction/convert_to_onnx.py --model-type pro

# 2. Build the Docker image for pro model
# Note: We specifically set platform to linux/amd64 since the container app runs on Azure
docker build --platform linux/amd64 \
  -t leaguedraftv2registrypro.azurecr.io/serve-model:latest \
  -f ./Dockerfile.pro \
  ./

# 3. Push the image to Azure Container Registry
docker push leaguedraftv2registrypro.azurecr.io/serve-model:latest

# 4. Update the container app
az containerapp update \
  --name leaguedraftv2inferencepro \
  --resource-group LeagueDraftv2 \
  --image leaguedraftv2registrypro.azurecr.io/serve-model:latest \
  --set-env-vars "API_KEY=secretref:api-key" \
  --revision-suffix $(date +%Y%m%d%H%M) 