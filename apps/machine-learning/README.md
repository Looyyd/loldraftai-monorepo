# /apps/machine-learning/README.md

# League of Legends Match Prediction Model

## Overview

This repository contains a machine learning system designed to predict various aspects of League of Legends matches, including win probability and in-game statistics, based on team compositions. The model uses a combination of champion embeddings and game state features to make these predictions.

## Architecture

### Data Pipeline

1. **Data Preparation** (`prepare_data.py`)
   - Downloads match data from Azure Blob Storage
   - Processes raw match data into training format
   - Handles categorical encoding and normalization
   - Creates train/test split
   - Filters out outliers and invalid games

### Model Architecture

The model uses a neural network architecture with several key components:

1. **Embeddings**:

   - Champion embeddings (learned representations for each champion)
   - Patch embeddings (meta changes across patches)
   - Champion-patch embeddings (champion-specific patch changes)
   - Categorical feature embeddings (queue type, ELO)

2. **Core Network**:

   - Multi-layer perceptron (MLP) with residual connections
   - Task-specific output heads for different predictions

3. **Training Features**:
   - Strategic masking during training to handle partial drafts
   - Multi-task learning for various predictions
   - Label smoothing for binary classification tasks

### Fine-tuning for Pro Play

The system includes a specialized fine-tuning pipeline (`train_pro.py`) for professional matches:

1. **Pro Data Adaptation**:

   - Uses pre-trained model as base
   - Gradually unfreezes layers during training
   - Specialized masking strategy for pro games
   - Separate validation for pro-specific metrics

2. **Model Adaptation**:
   - Handles new patches through embedding sliding/addition
   - Maintains model performance across patch changes
   - Supports online learning capabilities

## Key Features

- Win probability prediction
- Game duration prediction
- Per-player statistics at different timestamps
- Team-wide objective predictions
- Support for partial drafts
- Patch-aware predictions
- Pro play specialization
