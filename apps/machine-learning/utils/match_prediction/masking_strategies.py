from typing import Callable
import numpy as np


class LinearDecayDistribution:
    """
    Creates a linear decay distribution where probability of sampling 0 is decay_factor times
    higher than sampling max_value.
    """

    def __init__(self, max_value: int, decay_factor: float):
        self.max_value = max_value
        probabilities = np.linspace(decay_factor, 1, max_value + 1)
        self.probabilities = probabilities / probabilities.sum()

    def __call__(self) -> int:
        return np.random.choice(self.max_value + 1, p=self.probabilities)


class StrategicMaskingDistribution:
    """
    Probablilities are determined by my instinct, but seem abount right.
    Strategic masking with specific scenarios:
    - 40% chance to mask one full team (5 champions), to learn champion synergies
    - 0.5% chance to mask everything (10 champions), to have a baseline model output
    - 10% chance to mask nothing (0 champions), to learn complex champion interactions, with full visibility
    - 49.5% chance to mask 1-9 champions with linear distribution, to learn partial drafts
    """

    def __init__(self, decay_factor: float = 2.0):
        self.max_value = 10  # Total number of champions that can be masked

        # Initialize probabilities array
        self.probabilities = np.zeros(self.max_value + 1)

        # Set fixed scenario probabilities
        self.probabilities[0] = 0.10  # No masking
        self.probabilities[5] = 0.40  # One full team
        self.probabilities[10] = 0.005  # Everything masked

        # Distribute remaining probability (0.495) across 1-4, 6-9 using linear distribution
        remaining_indices = [i for i in range(1, 10) if i != 5]
        linear_probs = np.ones(len(remaining_indices))  # Equal probabilities
        linear_probs = linear_probs / linear_probs.sum() * 0.495

        # Assign the linearly distributed probabilities
        for idx, prob in zip(remaining_indices, linear_probs):
            self.probabilities[idx] = prob

        # Normalize to ensure sum is exactly 1
        self.probabilities = self.probabilities / self.probabilities.sum()

    def __call__(self) -> int:
        return np.random.choice(self.max_value + 1, p=self.probabilities)

    def get_distribution(self) -> np.ndarray:
        """Returns the probability distribution for debugging/visualization"""
        return self.probabilities


# Dictionary mapping strategy names to their classes
MASKING_STRATEGIES = {
    "linear_decay": LinearDecayDistribution,
    "strategic": StrategicMaskingDistribution,
}
