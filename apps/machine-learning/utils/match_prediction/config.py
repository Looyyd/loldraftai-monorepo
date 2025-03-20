# utils/match_prediction/config.py
import json
from typing import Callable
from utils.match_prediction.masking_strategies import MASKING_STRATEGIES


class TrainingConfig:
    def __init__(self):
        # Default values
        self.num_epochs = 25
        self.annealing_epoch = 10
        self.hidden_dims = [1536, 768, 384, 192]  # Both wider and one layer deeper
        self.dropout = 0.25  # Doubled dropout after adding positional embeddings
        self.learning_rate = 5e-4  # Lowered after positional embeddings were added
        self.embed_dim = 256  # seems optimal see experiments:
        # 128: https://wandb.ai/loyd-team/draftking/runs/hs7ocp6d?nw=nwuserloyd
        # 256: https://wandb.ai/loyd-team/draftking/runs/6w221kxa?nw=nwuserloyd
        # 1024: https://wandb.ai/loyd-team/draftking/runs/5eg66qlp?nw=nwuserloyd

        # weight decay didn't change much when training for a short time at 0.001, but for longer trianing runs, 0.01 might be better
        self.weight_decay = 0.001
        self.patch_reg_lambda = 0  # Weight for patch regularization loss
        self.champ_patch_reg_lambda = (
            0  # Regularization strength for champion+patch embeddings
        )
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

        # Add OneCycleLR parameters
        self.use_one_cycle_lr = True
        self.max_lr = self.learning_rate
        self.pct_start = 0.2  # 15% of training for warmup
        self.div_factor = 5  # initial_lr = max_lr/div_factor
        self.final_div_factor = 1e3  # final_lr = max_lr/(div_factor * final_div_factor)

        # Add new configuration parameters
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
