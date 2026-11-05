"""
data_loader.py - Simple Scenario Generator
==========================================
Generates simple day-by-day observations for our supply chain simulation.

In simple terms:
  - We have 3 possible real-world situations (Hidden States):
      0: Normal (Everything is running smoothly)
      1: Congestion (Port delays, minor backups)
      2: Crisis (Tariff shock, canal blockage)

  - But we cannot see the real situation directly!
    We only see operational signals (Observations):
      0: Cheap & On-Time (Good signs)
      1: Moderate Delay   (Medium signs)
      2: Expensive & Stuck (Bad signs)
"""

import numpy as np


class SupplyChainDataGenerator:
    """Creates simple simulation data to test our algorithms."""

    STATE_NAMES = {0: "Normal", 1: "Congestion", 2: "Crisis / Tariff Shock"}
    OBS_NAMES = {0: "Cheap & On-Time", 1: "Moderate Delay", 2: "Expensive & Stuck"}

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_scenario(self, scenario_type: str = "tariff_crisis", T: int = 15):
        """
        Creates a list of true states and what we observe each day for T days.
        """
        true_states = []
        observations = []

        for day in range(T):
            # 1. Determine what is actually happening in the world (Ground Truth)
            if scenario_type == "tariff_crisis":
                # First 5 days are Normal, next 6 days are Crisis, last 4 are recovering
                if day < 5:
                    state = 0  # Normal
                elif day < 11:
                    state = 2  # Crisis
                else:
                    state = 1  # Congestion / Recovery
            elif scenario_type == "canal_blockage":
                # Normal, then sudden canal blockage (congestion), then recovery
                if day < 4:
                    state = 0
                elif day < 10:
                    state = 1
                else:
                    state = 0
            else:  # baseline_normal
                state = 0

            # 2. Generate noisy sensor reading based on the state
            # If Normal: mostly clean signals (0)
            if state == 0:
                obs = self.rng.choice([0, 1, 2], p=[0.80, 0.15, 0.05])
            # If Congestion: mostly delay signals (1)
            elif state == 1:
                obs = self.rng.choice([0, 1, 2], p=[0.10, 0.70, 0.20])
            # If Crisis: mostly expensive & stuck signals (2)
            else:
                obs = self.rng.choice([0, 1, 2], p=[0.05, 0.25, 0.70])

            true_states.append(state)
            observations.append(obs)

        return {
            "T": T,
            "scenario": scenario_type,
            "true_states": true_states,
            "observations": observations,
        }
