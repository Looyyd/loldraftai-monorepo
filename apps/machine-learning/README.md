/apps/machine-learning/README.md

# League of Legends Match Prediction Model

## Architecture

### Data Pipeline

1. **Data Download** (`download_data.py`)

   - Downloads raw data from Azure bucket(which is were the data_collection exports match data)

2. **Data Preparation** (`prepare_data.py`)
   - Processes raw match data into training format
   - Handles categorical encoding(champions_ids, patches) and normalization
   - Creates train/test split
   - Filters out outliers and invalid games

### Model Architecture

The model uses a neural network architecture with several key components:

1. **Embeddings**:

   - Champion embeddings (learned representations for each champion)
   - Patch embeddings (meta changes across patches)
   - Champion-patch embeddings (champion-specific patch changes), this is small in dimension because could overfit easily, but in theory can slightly help represent patch specific champion changes
   - Categorical feature embeddings for queue_type(clash or solo queue), and elo(silver,gold etc.). The main reason a queue_type embedding is trained is because the same embedding is used to finetune the pro model(which uses a third queue_type for pro play).

2. **Core Network**:

   - Multi-layer perceptron (MLP) with residual connections
   - Task-specific output heads for different predictions(gold@15, win_prediction etc.)

3. **Training Features**:
   - Masking during training to handle partial drafts(masking is applied randomly, because draft order is not acessible through the API).
   - Multi-task learning for various predictions
   - Other training features and ways to induce bias to the model were tried, such as custom embedding initialization depending on champion class, regularization to ensure adjacent patches are close in embedding space or having way more auxillary tasks(such as damage dealt, baron kills, total_kills etc.). They can be seen in previous commits, but were removed in the simplified version. With a lot of training data(in the tens of millions) they are not beneficial, however with less data they did bring slight improvements.

### Fine-tuning for Pro Play

The system includes a specialized fine-tuning pipeline (`train_pro.py`) for professional matches, this model is available at https://loldraftai.com/pro-draft.

The finetune script uses the pre-trained model as base. A new queue_type embedding is trained for pro play, while other embeddings are frozen, the mlp layers are unfrozen. The model is finetune while keeping a significant portion of data from solo queue matches, to avoid catastrophic forgetting(which was observed if not using original data).

### Model serving

The model inference is done from an Azure docker container instance. Because the model is quite small and converted to onnx format(`convert_to_onnx.py`), it runs fast even on cpu inference. See `serve_model.py` and `serve-model-pro.py`. The models can easily be deployed with `./scripts/deploy-docker.sh` and `./scripts/deploy-docker-pro.sh`.

### Online learning/model updates

A script called `adapt_model.py` and `train.py` with the flag --continue_training can be used to update the model on a new patch without training from scratch. Can be useful if training is slow on some hardware, and was useful before training loop was optimized.
