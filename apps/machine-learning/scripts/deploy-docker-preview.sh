#!/bin/bash

# 1. Login to Azure Container Registry (if not already logged in)
az acr login --name leaguedraftv2registrypreview


# Build onnx image
python scripts/match-prediction/convert_to_onnx.py

# 2. Build the Docker image
# Note: We specifically set platform to linux/amd64 since the container app runs on Azure
docker build --platform linux/amd64 \
  -t leaguedraftv2registrypreview.azurecr.io/serve-model:latest \
  -f ./Dockerfile \
  ./


# 3. Push the image to Azure Container Registry
docker push leaguedraftv2registrypreview.azurecr.io/serve-model:latest

# 4. Update the container app (optional - if you want to deploy immediately)
az containerapp update \
  --name leaguedraftv2inference-preview \
  --resource-group LeagueDraftv2 \
  --image leaguedraftv2registrypreview.azurecr.io/serve-model:latest \
  --set-env-vars "API_KEY=secretref:api-key" \
  --revision-suffix $(date +%Y%m%d%H%M)
