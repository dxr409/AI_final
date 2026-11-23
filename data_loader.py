"""
data_loader.py - Supply Chain Scenario Generator & UI Presets
==============================================================
Generates day-by-day ground-truth states and noisy operational signals
for three realistic supply chain scenarios along Jebel Ali -> Hamburg:
  - all_clear:       Smooth sailing through Strait of Hormuz & Suez Canal
  - tariff_surge:    Customs backlog / port congestion after day 4
  - hormuz_blockage: Severe Strait of Hormuz crisis (days 4-10), then partial recovery

Hidden States:
  0: Normal
  1: Congestion
  2: Crisis

Observations:
  0: Cheap & On-Time
  1: Moderate Delay
  2: Expensive & Stuck
"""

from typing import Dict, Any
import numpy as np


PRESETS: Dict[str, Dict[str, Any]] = {
    "all_clear": {
        "label": "All Clear",
        "risk_pct": 5,
        "stock": 5,
        "tariff": 2,
        "description": "Calm seas and low tariffs: A* chooses the direct Hormuz + Suez maritime route.",
    },
    "tariff_surge": {
        "label": "Tariff Surge",
        "risk_pct": 40,
        "stock": 3,
        "tariff": 12,
        "description": "High import tariffs ($12/box) raise the expected tariff exposure of slow routes and curb MDP orders; Salalah + Suez wins initially, shifting to Air at extreme tariffs.",
    },
    "hormuz_blockage": {
        "label": "Hormuz Blockage",
        "risk_pct": 90,
        "stock": 3,
        "tariff": 2,
        "description": "Strait of Hormuz closure risk (90%): A* trucks around Hormuz to Salalah and sails around the Cape of Good Hope.",
    },
}


class SupplyChainDataGenerator:
    """Creates simulation scenarios and noisy observations."""

    STATE_NAMES = {0: "Normal", 1: "Congestion", 2: "Crisis"}
    OBS_NAMES = {0: "Cheap & On-Time", 1: "Moderate Delay", 2: "Expensive & Stuck"}

    # Matches SimpleHMM.B emission probabilities
    EMISSION_PROBS = {
        0: [0.80, 0.15, 0.05],
        1: [0.10, 0.70, 0.20],
        2: [0.05, 0.25, 0.70],
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_scenario(
        self, scenario_type: str = "hormuz_blockage", T: int = 15
    ) -> Dict[str, Any]:
        """
        Generates T days of ground-truth states and noisy operational observations.
        """
        rng = np.random.default_rng(self.seed)
        true_states = []
        observations = []

        for day in range(T):
            if scenario_type == "all_clear":
                state = 0
            elif scenario_type == "tariff_surge":
                state = 0 if day < 4 else 1
            elif scenario_type == "hormuz_blockage":
                if day < 4:
                    state = 0
                elif day <= 10:
                    state = 2
                else:
                    state = 1
            else:
                raise ValueError(f"Unknown scenario_type: {scenario_type}")

            obs = int(rng.choice([0, 1, 2], p=self.EMISSION_PROBS[state]))
            true_states.append(state)
            observations.append(obs)

        return {
            "T": T,
            "scenario": scenario_type,
            "true_states": true_states,
            "observations": observations,
        }
