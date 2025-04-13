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
    Strategic masking with specific scenarios.
    """

    def __init__(self, decay_factor: float = 2.0):
        self.max_value = 10  # Total number of champions that can be masked

        # Initialize probabilities array
        self.probabilities = np.zeros(self.max_value + 1)

        # Set fixed scenario probabilities
        remaining_prob = 1
        self.probabilities[0] = 0.20  # No masking
        remaining_prob -= self.probabilities[0]
        self.probabilities[10] = 0.01  # Everything masked
        remaining_prob -= self.probabilities[10]

        # In master tier, the model seems undertrained when only seeing  1 to 3 champions.
        # For this reason, we increase the probabilities to only have 1 to 3 champions visible.
        # To force the model to learn patterns when only 1 to 3 champions are visible
        self.probabilities[9] = 0.15
        remaining_prob -= self.probabilities[9]
        self.probabilities[8] = 0.15
        remaining_prob -= self.probabilities[8]
        self.probabilities[7] = 0.15
        remaining_prob -= self.probabilities[7]

        # Distribute remaining probability across remaining indices using linear distribution
        remaining_indices = [i for i in range(1, 7)]
        linear_probs = np.ones(len(remaining_indices))  # Equal probabilities
        linear_probs = linear_probs / linear_probs.sum() * remaining_prob

        # Assign the linearly distributed probabilities
        # Assign the linearly distributed probabilities
        for idx, prob in zip(remaining_indices, linear_probs):
            self.probabilities[idx] = prob

        # Normalize to ensure sum is exactly 1
        self.probabilities = self.probabilities / self.probabilities.sum()

    def __call__(self) -> int:
        # TODO: should maybe return indices, to allow masking of 1 full team, instead of random 5 champions
        return np.random.choice(self.max_value + 1, p=self.probabilities)

    def get_distribution(self) -> np.ndarray:
        """Returns the probability distribution for debugging/visualization"""
        return self.probabilities


# Dictionary mapping strategy names to their classes
MASKING_STRATEGIES = {
    "linear_decay": LinearDecayDistribution,
    "strategic": StrategicMaskingDistribution,
}
