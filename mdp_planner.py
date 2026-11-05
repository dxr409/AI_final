"""
mdp_planner.py - Markov Decision Process (MDP) Inventory Optimizer
==================================================================
Topic: Sequential Decision-Making under Uncertainty

WHAT THIS DOES IN SIMPLE WORDS:
  You manage a warehouse that holds up to 8 boxes of goods.
  Every day, customers want to buy 2 boxes.
  
  You must decide: "How many boxes should I order today?"
  Your choices (Actions): Order 0, 2, 4, or 6 boxes.
  
  The twist:
    - If world is Normal: 100% of your order arrives safely.
    - If world is in Congestion: Only 50% arrives (half gets delayed).
    - If world is in Crisis: 0% arrives (stuck at customs / tariffs).

COSTS YOU WANT TO MINIMIZE:
  1. Ordering Cost: $4 per box ordered.
  2. Holding Cost: $1 per unsold box stored in warehouse overnight.
  3. Stockout Penalty: $30 per box if customers want to buy and warehouse is empty!
  4. Tariff Fee: Extra $4 per box if ordering during a Crisis.

ALGORITHM: Bellman Value Iteration
  We compute the expected total cost for each (Inventory, RiskState) pair,
  and find the best order action that minimizes:
    Total Cost = Today's Immediate Cost + (0.90 * Expected Future Cost Tomorrow)
"""

import numpy as np


class SimpleInventoryMDP:
    """Solves for the optimal inventory reordering policy using Value Iteration."""

    def __init__(self, max_inventory: int = 8, discount_factor: float = 0.90):
        self.max_inventory = max_inventory
        self.gamma = discount_factor

        # Order choices available to the manager
        self.actions = [0, 2, 4, 6]

        # Daily customer demand (averages 2 boxes per day)
        self.daily_demand = 2

        # Costs
        self.cost_per_item = 4.0
        self.holding_cost_per_item = 1.0
        self.stockout_penalty = 30.0
        self.tariff_extra_cost = 4.0

        # Delivery arrival rate based on corridor state:
        # Normal -> 1.0 (100% arrives), Congestion -> 0.5 (50%), Crisis -> 0.0 (0%)
        self.delivery_success = {0: 1.0, 1: 0.5, 2: 0.0}

        # Transition probabilities between states (from HMM transition matrix)
        self.state_transitions = [
            [0.85, 0.12, 0.03],  # From Normal
            [0.20, 0.70, 0.10],  # From Congestion
            [0.05, 0.25, 0.70],  # From Crisis
        ]

        # Value table V[inventory, state] and Policy table policy[inventory, state]
        self.V = np.zeros((self.max_inventory + 1, 3))
        self.policy = np.zeros((self.max_inventory + 1, 3), dtype=int)

    def calculate_cost(self, current_inv: int, state: int, action: int) -> float:
        """Calculates today's immediate cost for taking an action."""
        # 1. Purchase cost
        purchase_cost = self.cost_per_item * action
        if state == 2:  # Tariff crisis surcharge
            purchase_cost += self.tariff_extra_cost * action

        # 2. How many items actually arrive at the warehouse today?
        arrived = int(round(action * self.delivery_success[state]))
        stock_available = current_inv + arrived

        # 3. Customer demand fulfillment
        if stock_available >= self.daily_demand:
            unsold = stock_available - self.daily_demand
            stockout_items = 0
        else:
            unsold = 0
            stockout_items = self.daily_demand - stock_available

        # 4. Warehouse holding cost & stockout penalties
        ending_inv = min(self.max_inventory, unsold)
        holding_cost = self.holding_cost_per_item * ending_inv
        penalty_cost = self.stockout_penalty * stockout_items

        return purchase_cost + holding_cost + penalty_cost

    def get_next_inventory(self, current_inv: int, state: int, action: int) -> int:
        """Determines tomorrow's starting inventory level."""
        arrived = int(round(action * self.delivery_success[state]))
        stock_available = current_inv + arrived
        ending_inv = max(0, stock_available - self.daily_demand)
        return min(self.max_inventory, ending_inv)

    def solve_value_iteration(self, max_iterations: int = 100, tolerance: float = 0.01):
        """
        Solves the Bellman equation by repeating until the values stop changing:
          V(s) = min_a [ Cost(s, a) + gamma * Expected_V(next_s) ]
        """
        for iteration in range(max_iterations):
            max_change = 0.0
            new_V = np.zeros_like(self.V)

            # Loop through all inventory levels (0 to 8)
            for inv in range(self.max_inventory + 1):
                # Loop through all 3 states (Normal, Congestion, Crisis)
                for s in range(3):
                    costs_for_each_action = []

                    for a in self.actions:
                        # 1. Today's immediate cost
                        today_cost = self.calculate_cost(inv, s, a)

                        # 2. Next inventory level
                        next_inv = self.get_next_inventory(inv, s, a)

                        # 3. Tomorrow's expected future cost across possible tomorrow states
                        future_cost = 0.0
                        for next_s in range(3):
                            prob = self.state_transitions[s][next_s]
                            future_cost += prob * self.V[next_inv, next_s]

                        total_cost = today_cost + self.gamma * future_cost
                        costs_for_each_action.append(total_cost)

                    # Pick the action that gives the lowest total cost!
                    best_cost = min(costs_for_each_action)
                    best_action_index = int(np.argmin(costs_for_each_action))

                    new_V[inv, s] = best_cost
                    self.policy[inv, s] = self.actions[best_action_index]

                    change = abs(new_V[inv, s] - self.V[inv, s])
                    if change > max_change:
                        max_change = change

            self.V = new_V
            if max_change < tolerance:
                break

        return self.policy, self.V

    def get_urgency_score(self, current_inv: int) -> float:
        """
        Passes a simple Urgency Score between 0.0 and 1.0 to Algorithm 3 (A* Router).
        - If inventory is 0: Urgency is 1.0 (CRITICAL! Need goods immediately!)
        - If inventory is 1: Urgency is 0.7 (HIGH)
        - If inventory is 2: Urgency is 0.4 (MEDIUM)
        - If inventory >= 3: Urgency is 0.0 (LOW, plenty in stock)
        """
        if current_inv == 0:
            return 1.0
        elif current_inv == 1:
            return 0.7
        elif current_inv == 2:
            return 0.4
        else:
            return 0.0
