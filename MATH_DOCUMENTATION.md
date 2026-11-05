# Mathematical Foundations (Simplified Guide)
## Supply Chain Vulnerability & Tariff Disruption Advisory

This document explains the mathematical formulas used in this project in clear, simple steps that any student or teacher can easily follow.

---

## 1. Algorithm 1: Hidden Markov Model (HMM)
**Course Topic: Probabilistic Reasoning / Temporal Models**

### What It Does
We want to guess what state the world is in, even though we cannot observe it directly.

- **3 Hidden States ($S$)**:
  - $s_0$: Normal
  - $s_1$: Congestion (Port Delays)
  - $s_2$: Crisis (Tariff / Canal Shock)

- **3 Observations ($Y$)**:
  - $y_0$: Cheap & On-Time
  - $y_1$: Moderate Delay
  - $y_2$: Expensive & Stuck

### The Two Matrices
1. **Transition Matrix $A$ (How tomorrow changes from today)**:
   $$A_{i, j} = P(S_{t+1} = s_j \mid S_t = s_i)$$
   Example: If today is Normal, there is an 85% chance tomorrow remains Normal.

2. **Emission Matrix $B$ (What we see in each state)**:
   $$B_{j, k} = P(Y_t = y_k \mid S_t = s_j)$$
   Example: If the world is in Crisis, there is a 70% chance we observe "Expensive & Stuck".

### The Forward Filtering Formula (2 Simple Steps)
Every time we get a new observation $Y_t$:
1. **Prediction**:
   $$\text{Predicted State} = \text{Belief}_{t-1} \times A$$
2. **Measurement Update & Normalization**:
   $$\text{Belief}_t = \frac{\text{Predicted State} \times B[:, Y_t]}{\text{Sum of all probabilities}}$$

The resulting vector $[\text{P(Normal)}, \text{P(Congestion)}, \text{P(Crisis)}]$ always sums up to $1.0$ (100%).

---

## 2. Algorithm 2: Markov Decision Process (MDP)
**Course Topic: Sequential Decision-Making under Uncertainty**

### What It Does
Decides how many boxes of laptops to order for a warehouse each day.

- **State**: $(\text{Inventory}, \text{Risk State})$
  - $\text{Inventory} \in \{0, 1, 2, \dots, 8\}$ boxes.
  - $\text{Risk State} \in \{\text{Normal}, \text{Congestion}, \text{Crisis}\}$ (from the HMM).
- **Actions**: Order $0, 2, 4,$ or $6$ boxes.
- **Customer Demand**: Customers buy $2$ boxes every day.

### Costs to Minimize
$$\text{Cost} = \text{Order Cost} + \text{Holding Cost} + \text{Stockout Penalty} + \text{Tariff Fee}$$
- Order Cost = $\$4 \times \text{boxes ordered}$
- Holding Cost = $\$1 \times \text{unsold boxes left in warehouse}$
- Stockout Penalty = $\$30 \times \text{customers turned away if warehouse is empty!}$
- Tariff Fee = $\$4 \times \text{boxes ordered}$ (applied only in Crisis state)

### Bellman Value Iteration Formula
$$\text{Total Value}(s) = \min_{a} \Big[ \text{Immediate Cost}(s, a) + 0.90 \times \text{Expected Future Cost}(\text{next } s) \Big]$$
- We start with all values at 0.
- We update the table for every inventory level and state until the numbers stop changing (convergence).
- The action that gives the smallest cost becomes our **optimal policy** $\pi^*(I, S)$.

---

## 3. Algorithm 3: Dynamic A* Search
**Course Topic: Classical Search & Heuristics**

### What It Does
Finds the cheapest route from **Shanghai** to **Hamburg** across 4 multimodal corridor choices:
1. **Sea via Suez Canal**: Very cheap (\$100), but takes 20 days and passes through the vulnerable Suez Canal.
2. **Sea via Cape of Good Hope**: Cheap (\$160), takes 30 days, but completely avoids the Suez Canal.
3. **Express Train**: Medium price (\$200), fast (10 days), travels overland.
4. **Air Cargo**: Expensive (\$400), ultra-fast (2 days).

### The Dynamic Cost Formula (Connecting HMM + MDP to A*)
$$\text{Edge Cost} = \text{Base Cost} + \text{Disruption Penalty} + \text{Urgency Penalty}$$
- **Disruption Penalty** = $\text{Base Cost} \times \text{HMM Disruption Probability} \times \text{Canal Risk Factor}$
  *(If HMM detects high crisis probability, the Suez route becomes expensive!)*
- **Urgency Penalty** = $\text{MDP Urgency Score} \times 12.0 \times \text{Transit Days}$
  *(If MDP detects inventory is 0, slow routes with 20-30 days are heavily penalized!)*

### The A* Heuristic
$$h(n) = \text{Straight Line Distance to Hamburg} \times 8.0$$
- **Admissibility**: Because a straight line is the shortest possible path between two points, $h(n)$ **never overestimates** the real cost ($h(n) \le h^*(n)$).
- Therefore, A* is mathematically guaranteed to find the true optimal route.
