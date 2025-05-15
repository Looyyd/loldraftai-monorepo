# utils/match_prediction/config.py
import json
from typing import Callable
from utils.match_prediction.masking_strategies import MASKING_STRATEGIES


class TrainingConfig:
    def __init__(self, continue_training: bool = False):
        # Default values
        self.num_epochs = 50
        self.annealing_epoch = 10
        # Reduce hidden dimensions to maintain similar parameter count
        self.hidden_dims = [192, 128, 96, 64]  # Smaller hidden dimensions
        self.dropout = 0
        self.champion_patch_embed_dim = 4  # Small dimension to avoid overfitting
        self.champion_embed_dim = (
            128 - self.champion_patch_embed_dim
        )  # Reduced from 256
        self.queue_type_embed_dim = 32  # Reduced from 64
        self.patch_embed_dim = 64  # Reduced from 128
        self.elo_embed_dim = 32  # Reduced from 64

        # New Qwen3-inspired model parameters
        self.use_qwen3_architecture = True
        self.num_attention_heads = 4  # Reduced from 8
        self.num_kv_heads = 2  # Number of key-value heads for grouped query attention
        self.head_dim = (
            None  # If None, will be computed as hidden_size // num_attention_heads
        )
        self.intermediate_size_factor = 2  # Reduced from 4
        self.max_position_embeddings = (
            13  # Maximum sequence length for positional embeddings
        )
        self.rms_norm_eps = 1e-6  # Epsilon for RMSNorm
        self.use_sliding_window = False  # Whether to use sliding window attention

        # Sequence modeling parameters
        self.use_sequence_modeling = True  # Model input as a sequence of tokens
        self.use_token_type_embeddings = (
            True  # Use token type embeddings to distinguish different token types
        )
        self.pooling_method = (
            "mean"  # How to pool sequence outputs: mean, cls, or attention
        )

        # weight decay didn't change much when training for a short time at 0.001, but for longer trianing runs, 0.01 might be better
        self.weight_decay = 0.05
        self.elo_reg_lambda = 0  # Weight for Elo regularization loss
        self.patch_reg_lambda = 0  # Weight for patch regularization loss
        self.champ_patch_reg_lambda = 0.0
        self.max_grad_norm = 1.0  # because has loss spikes after adding pos embeddings
        self.accumulation_steps = 1
        self.masking_strategy = {
            "name": "strategic",
            "params": {"decay_factor": 2.0},
        }

        self.calculate_val_loss = True
        self.calculate_val_win_prediction_only = True
        self.log_wandb = True
        self.debug = False

        if continue_training:
            # Configuration for continued training (online learning)
            self.learning_rate = 4e-4  # Lower LR for continued training (was 4e-4)
            self.use_one_cycle_lr = False  # No one-cycle scheduler
        else:
            # Regular training configuration
            self.learning_rate = 1e-3  # Reduced from 5e-3 to a more appropriate rate for transformer models
            self.use_one_cycle_lr = True
            self.max_lr = self.learning_rate
            self.pct_start = 0.2
            self.div_factor = 10
            self.final_div_factor = 3e4  # Add new configuration parameters

        self.validation_interval = 1  # Run validation every N epochs
        self.dataset_fraction = 1.0  # Use full dataset by default

        self.track_subset_val_losses = (
            True  # Track validation metrics by patch, ELO, and champion ID
        )

    def update_from_json(self, json_file: str):
        with open(json_file, "r") as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter '{key}'")

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    def to_dict(self) -> dict:
        return {key: value for key, value in vars(self).items() if key != "log_wandb"}

    def get_masking_function(self) -> Callable[[], int]:
        """Returns a function that generates number of champions to mask"""
        strategy = MASKING_STRATEGIES[self.masking_strategy["name"]]
        return strategy(**self.masking_strategy["params"])
