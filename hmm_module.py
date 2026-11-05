"""
hmm_module.py - Hidden Markov Model (HMM)
=========================================
Topic: Probabilistic Reasoning & Temporal Models

WHAT THIS DOES IN SIMPLE WORDS:
  We want to know: "Is the shipping corridor in Normal, Congestion, or Crisis state?"
  We cannot see the true state directly, but each day we observe shipping signals:
    0: Cheap & On-Time
    1: Moderate Delay
    2: Expensive & Stuck

  The Forward Algorithm updates our belief vector [P(Normal), P(Congestion), P(Crisis)]
  after each new observation using basic probability.

THE TWO MATRICES:
  1. Transition Matrix A (How the world changes day-to-day):
       A[i][j] = Probability of going from State i today to State j tomorrow.
  2. Emission Matrix B (What we see in each state):
       B[j][k] = Probability of seeing Observation k when in State j.

THE 2-LINE UPDATE RULE:
  Step 1: predicted = previous_belief @ A
  Step 2: unnormalized = predicted * B[:, observation]
  Step 3: belief = unnormalized / sum(unnormalized)   # makes it add up to 100%
"""

import numpy as np


class SimpleHMM:
    """Hidden Markov Model for estimating shipping disruption risk."""

    STATE_NAMES = ["Normal", "Congestion", "Crisis"]

    def __init__(self):
        # Initial belief: We assume day 1 starts mostly Normal
        # [80% Normal, 15% Congestion, 5% Crisis]
        self.initial_belief = np.array([0.80, 0.15, 0.05])

        # Transition Matrix A: P(Tomorrow | Today)
        # Rows = Today's state, Columns = Tomorrow's state
        self.A = np.array([
            # Tomorrow: Normal  Congestion  Crisis
            [0.85, 0.12, 0.03],  # Today is Normal
            [0.20, 0.70, 0.10],  # Today is Congestion
            [0.05, 0.25, 0.70],  # Today is Crisis
        ])

        # Emission Matrix B: P(Observation | Current State)
        # Rows = Hidden state, Columns = Observation (0: Good, 1: Delayed, 2: Stuck)
        self.B = np.array([
            # Obs: Good  Delayed  Stuck
            [0.80, 0.15, 0.05],  # In Normal state
            [0.10, 0.70, 0.20],  # In Congestion state
            [0.05, 0.25, 0.70],  # In Crisis state
        ])

        self.current_belief = self.initial_belief.copy()

    def reset(self):
        """Reset belief to the beginning of the simulation."""
        self.current_belief = self.initial_belief.copy()

    def update_belief(self, observation: int) -> np.ndarray:
        """
        Runs one step of the Forward Algorithm for a new observation.
        Returns: belief vector [P(Normal), P(Congestion), P(Crisis)].
        """
        # Step 1: Predict what state we might be in today based on yesterday's belief
        # (Matrix multiplication: belief vector times transition matrix)
        predicted_state = self.current_belief @ self.A

        # Step 2: Multiply by the chance of seeing what we actually observed
        likelihood = self.B[:, observation]
        updated_belief = predicted_state * likelihood

        # Step 3: Normalize so the 3 probabilities sum up to 1.0 (100%)
        total = np.sum(updated_belief)
        if total > 0:
            self.current_belief = updated_belief / total
        else:
            self.current_belief = np.array([0.33, 0.33, 0.34])

        return self.current_belief

    def get_disruption_probability(self) -> float:
        """
        Single number between 0.0 (safe) and 1.0 (disrupted) passed to A* router.
        Combines congestion (weighted 0.4) and crisis (weighted 1.0).
        """
        p_normal = self.current_belief[0]
        p_congestion = self.current_belief[1]
        p_crisis = self.current_belief[2]
        return float(0.4 * p_congestion + 1.0 * p_crisis)

    def get_most_likely_state(self) -> int:
        """Returns the state with the highest probability (0, 1, or 2)."""
        return int(np.argmax(self.current_belief))
