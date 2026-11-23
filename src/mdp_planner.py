"""
mdp_planner.py - Markov Decision Process (MDP) Inventory Optimizer
==================================================================
Topic: Sequential Decision-Making under Uncertainty

WHAT THIS DOES:
  Manages a warehouse holding up to 8 boxes of electronics imported from Jebel Ali to Hamburg.
  Daily customer demand averages 2 boxes.

  States: (Inventory I in {0..8}, Corridor Risk State s in {Normal, Congestion, Crisis})
  Actions: Order a in {0, 2, 4, 6} boxes.

  Delivery arrival rate along vulnerable corridors:
    - Normal (s=0):     100% arrives
    - Congestion (s=1):  50% arrives
    - Crisis (s=2):      25% arrives (severe chokepoint blockage)

COSTS TO MINIMIZE:
  - Procurement + Tariff: ($4 + tariff_per_box) per box ordered (known policy input)
  - Holding Cost:         $1 per box remaining at end of day
  - Stockout Penalty:     $30 per unfulfilled unit of daily demand

ALGORITHM: Bellman Value Iteration
  V(I, s) = min_a [ immediate_cost(I, s, a) + gamma * sum_{s'} P(s'|s) V(I', s') ]
"""

from typing import Any, Dict, Tuple
import numpy as np


class SimpleInventoryMDP:
    """Solves for the optimal inventory reordering policy using Bellman Value Iteration."""

    def __init__(
        self,
        max_inventory: int = 8,
        discount_factor: float = 0.90,
        tariff_per_box: float = 2.0,
    ):
        self.max_inventory = max_inventory
        self.gamma = discount_factor
        self.tariff_per_box = float(tariff_per_box)

        self.actions = [0, 2, 4, 6]
        self.daily_demand = 2

        # Cost parameters (USD)
        self.cost_per_item = 4.0
        self.holding_cost_per_item = 1.0
        self.stockout_penalty = 30.0

        # Delivery arrival rate based on corridor state:
        # Normal -> 1.0 (100%), Congestion -> 0.5 (50%), Crisis -> 0.25 (25%)
        self.delivery_success = {0: 1.0, 1: 0.5, 2: 0.25}

        # State transition matrix P(s' | s), matching SimpleHMM.A
        self.state_transitions = [
            [0.85, 0.12, 0.03],  # From Normal
            [0.20, 0.70, 0.10],  # From Congestion
            [0.05, 0.25, 0.70],  # From Crisis
        ]

        self.V = np.zeros((self.max_inventory + 1, 3))
        self.policy = np.zeros((self.max_inventory + 1, 3), dtype=int)
        self.iterations_used = 0

    def get_next_inventory(self, current_inv: int, state: int, action: int) -> int:
        """Determines tomorrow's starting inventory level after arrivals and demand."""
        arrived = int(round(action * self.delivery_success[state]))
        stock_available = current_inv + arrived
        sold = min(self.daily_demand, stock_available)
        return min(self.max_inventory, stock_available - sold)

    def q_value_breakdown(
        self, current_inv: int, state: int, action: int
    ) -> Dict[str, Any]:
        """
        Computes full Bellman Q(I, s, a) cost breakdown for a single (inventory, state, action).
        """
        order_cost = (self.cost_per_item + self.tariff_per_box) * action
        arrived = int(round(action * self.delivery_success[state]))
        stock_available = current_inv + arrived
        sold = min(self.daily_demand, stock_available)
        shortage = self.daily_demand - sold
        next_inv = min(self.max_inventory, stock_available - sold)

        holding = self.holding_cost_per_item * next_inv
        stockout_cost = self.stockout_penalty * shortage
        immediate = order_cost + holding + stockout_cost

        future = self.gamma * sum(
            self.state_transitions[state][ns] * float(self.V[next_inv, ns])
            for ns in range(3)
        )
        total = immediate + future

        return {
            "action": action,
            "order_cost": float(order_cost),
            "arrived": arrived,
            "next_inv": next_inv,
            "holding": float(holding),
            "shortage": shortage,
            "stockout_cost": float(stockout_cost),
            "immediate": float(immediate),
            "future": float(future),
            "total": float(total),
        }

    def calculate_cost(self, current_inv: int, state: int, action: int) -> float:
        """Calculates today's immediate cost for taking an action."""
        return self.q_value_breakdown(current_inv, state, action)["immediate"]

    def solve_value_iteration(
        self, max_iterations: int = 500, tolerance: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the Bellman optimality equation via Value Iteration until convergence.
        """
        self.V = np.zeros((self.max_inventory + 1, 3))
        self.policy = np.zeros((self.max_inventory + 1, 3), dtype=int)
        self.iterations_used = 0

        for iteration in range(1, max_iterations + 1):
            max_change = 0.0
            new_V = np.zeros_like(self.V)

            for inv in range(self.max_inventory + 1):
                for s in range(3):
                    q_values = [
                        self.q_value_breakdown(inv, s, a)["total"] for a in self.actions
                    ]
                    best_idx = int(np.argmin(q_values))
                    best_cost = q_values[best_idx]

                    new_V[inv, s] = best_cost
                    self.policy[inv, s] = self.actions[best_idx]

                    change = abs(new_V[inv, s] - self.V[inv, s])
                    if change > max_change:
                        max_change = change

            self.V = new_V
            self.iterations_used = iteration
            if max_change < tolerance:
                break

        return self.policy, self.V

    def get_urgency_score(self, current_inv: int) -> float:
        """
        Passes an Urgency Score between 0.0 and 1.0 to Dynamic A* Search:
          - 0 boxes -> 1.0 (CRITICAL)
          - 1 box   -> 0.7 (HIGH)
          - 2 boxes -> 0.4 (MEDIUM)
          - >= 3    -> 0.0 (LOW)
        """
        if current_inv == 0:
            return 1.0
        elif current_inv == 1:
            return 0.7
        elif current_inv == 2:
            return 0.4
        else:
            return 0.0
