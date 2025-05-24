# utils/match_prediction/config.py
import json
from typing import Callable
from utils.match_prediction.masking_strategies import MASKING_STRATEGIES


class TrainingConfig:
    def __init__(self, continue_training: bool = False):
        # Default values
        self.num_epochs = 50
        # Deeper network with more layers
        self.hidden_dims = [
            1024,
            512,
            512,
            512,
            512,
            256,
            256,
            256,
            256,
            128,
            64,
        ]
        self.dropout = 0.5
        self.champion_patch_embed_dim = (
            2  # Needs to be small to avoid overfit, was 4 before and that's fine.
        )
        self.champion_embed_dim = (
            128 - self.champion_patch_embed_dim
        )  # Reduced from 256
        self.queue_type_embed_dim = 32
        self.patch_embed_dim = 64
        self.elo_embed_dim = 32

        self.weight_decay = 0.05
        self.max_grad_norm = 1.0
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
            self.learning_rate = 4e-4  # Lower LR for continued training
            self.use_one_cycle_lr = False  # No one-cycle scheduler
        else:
            # Regular training configuration
            self.learning_rate = 8e-4
            self.use_one_cycle_lr = True
            self.max_lr = self.learning_rate
            self.pct_start = 0.2
            self.div_factor = 10
            self.final_div_factor = 3e4

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
