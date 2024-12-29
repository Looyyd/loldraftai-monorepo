import json
from typing import Callable

from utils.match_prediction.masking_strategies import MASKING_STRATEGIES


class TrainingConfig:
    def __init__(self):
        # Default values
        self.num_epochs = 25
        self.embed_dim = 64
        self.num_heads = 8
        self.num_transformer_layers = 2
        self.dropout = 0.1
        # weight decay didn't change much when training for a short time at 0.001, but for longer trianing runs, 0.01 might be better
        self.weight_decay = 0.01
        self.learning_rate = 5e-2
        self.max_grad_norm = 1.0
        self.accumulation_steps = 1
        self.masking_strategy = {
            "name": "strategic",
            "params": {"decay_factor": 2.0},
        }

        self.calculate_val_loss = True
        self.calculate_val_win_prediction_only = True
        self.log_wandb = True

        # Add OneCycleLR parameters
        self.use_one_cycle_lr = True
        self.max_lr = 2e-3
        self.pct_start = 0.3  # 30% of training for warmup
        self.div_factor = 25.0  # initial_lr = max_lr/div_factor
        self.final_div_factor = 1e4  # final_lr = max_lr/(div_factor * final_div_factor)

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
