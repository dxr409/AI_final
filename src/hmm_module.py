"""
hmm_module.py - Hidden Markov Model (HMM)
=========================================
Topic: Probabilistic Reasoning & Temporal Models

Infers the hidden shipping corridor state (Normal, Congestion, Crisis)
from noisy daily operational signals:
  0: Cheap & On-Time
  1: Moderate Delay
  2: Expensive & Stuck

Runs exact Forward Filtering:
  1. predicted = prev_belief @ A
  2. unnormalized = predicted * B[:, observation]
  3. posterior = unnormalized / sum(unnormalized)
"""

from typing import Any, Dict
import numpy as np


def risk_to_state(p: float) -> int:
    """
    Maps scalar disruption risk p = 0.4 * P(Congestion) + 1.0 * P(Crisis)
    to discrete MDP risk states using midpoints between pure-state p values (0.0, 0.4, 1.0).
    Used when driving the interactive dashboard directly from the risk slider.
    """
    if p < 0.2:
        return 0
    elif p < 0.7:
        return 1
    else:
        return 2


class SimpleHMM:
    """Hidden Markov Model for estimating shipping disruption risk."""

    STATE_NAMES = ["Normal", "Congestion", "Crisis"]
    OBS_NAMES = ["Cheap & On-Time", "Moderate Delay", "Expensive & Stuck"]

    def __init__(self):
        # Initial belief: [80% Normal, 15% Congestion, 5% Crisis]
        self.initial_belief = np.array([0.80, 0.15, 0.05], dtype=float)

        # Transition Matrix A: P(Tomorrow | Today)
        self.A = np.array(
            [
                # Tomorrow: Normal  Congestion  Crisis
                [0.85, 0.12, 0.03],  # Today is Normal
                [0.20, 0.70, 0.10],  # Today is Congestion
                [0.05, 0.25, 0.70],  # Today is Crisis
            ],
            dtype=float,
        )

        # Emission Matrix B: P(Observation | Current State)
        self.B = np.array(
            [
                # Obs: Good  Delayed  Stuck
                [0.80, 0.15, 0.05],  # In Normal state
                [0.10, 0.70, 0.20],  # In Congestion state
                [0.05, 0.25, 0.70],  # In Crisis state
            ],
            dtype=float,
        )

        self.current_belief = self.initial_belief.copy()

    def reset(self) -> None:
        """Reset belief to the beginning of the simulation."""
        self.current_belief = self.initial_belief.copy()

    def forward_step_details(
        self, prev_belief: np.ndarray, observation: int
    ) -> Dict[str, Any]:
        """
        Pure forward filtering step returning all intermediate vectors:
        predicted, likelihood, unnormalized, Z, and posterior.
        """
        prev = np.asarray(prev_belief, dtype=float)
        predicted = prev @ self.A
        likelihood = self.B[:, observation].copy()
        unnormalized = predicted * likelihood
        Z = float(np.sum(unnormalized))
        if Z > 0.0:
            posterior = unnormalized / Z
        else:
            posterior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
            Z = 1.0

        return {
            "predicted": predicted,
            "likelihood": likelihood,
            "unnormalized": unnormalized,
            "Z": Z,
            "posterior": posterior,
        }

    def update_belief(self, observation: int) -> np.ndarray:
        """Runs one step of the Forward Algorithm and updates current_belief."""
        details = self.forward_step_details(self.current_belief, observation)
        self.current_belief = details["posterior"].copy()
        return self.current_belief

    def get_disruption_probability(self) -> float:
        """
        Single scalar between 0.0 (safe) and 1.0 (disrupted) passed to A* router:
          p = 0.4 * P(Congestion) + 1.0 * P(Crisis)
        """
        p_congestion = float(self.current_belief[1])
        p_crisis = float(self.current_belief[2])
        return float(0.4 * p_congestion + 1.0 * p_crisis)

    def get_most_likely_state(self) -> int:
        """Returns the state with the highest posterior probability (0, 1, or 2)."""
        return int(np.argmax(self.current_belief))
