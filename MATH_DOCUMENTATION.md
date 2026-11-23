# Mathematical Foundations (Simplified Guide)
## Supply Chain Vulnerability & Tariff Disruption Advisory (Jebel Ali → Hamburg)

This document explains the mathematical formulas implemented in this project clearly and concisely.

---

## 1. Algorithm 1: Hidden Markov Model (HMM)
**Course Topic: Probabilistic Reasoning / Temporal Models**

### What It Does
We want to infer the hidden state of the shipping corridor from noisy daily operational reports.

- **3 Hidden States ($S$)**:
  - $s_0$: Normal
  - $s_1$: Congestion (Port / Customs Delays)
  - $s_2$: Crisis (Strait of Hormuz / Chokepoint Shock)

- **3 Observations ($Y$)**:
  - $y_0$: Cheap & On-Time
  - $y_1$: Moderate Delay
  - $y_2$: Expensive & Stuck

### The Two Matrices
1. **Transition Matrix $A$**:
   $$A_{i, j} = P(S_{t+1} = s_j \mid S_t = s_i)$$
2. **Emission Matrix $B$**:
   $$B_{j, k} = P(Y_t = y_k \mid S_t = s_j)$$

### Forward Filtering Step
For each new daily observation $y_t$:
1. **Predict**: $\hat{b}_t(j) = \sum_i b_{t-1}(i) \cdot A_{i,j}$
2. **Update & Normalize**:
   $$b_t(j) = \frac{B[j, y_t] \cdot \hat{b}_t(j)}{Z}, \quad Z = \sum_{k=0}^{2} B[k, y_t] \cdot \hat{b}_t(k)$$
3. **Scalar Disruption Risk ($p$) passed to A\***:
   $$p = 0.4 \cdot b_t(\text{Congestion}) + 1.0 \cdot b_t(\text{Crisis})$$

---

## 2. Algorithm 2: Markov Decision Process (MDP)
**Course Topic: Sequential Decision-Making under Uncertainty**

### What It Does
Decides how many boxes of laptops (`0, 2, 4, or 6`) to order each day for a warehouse with capacity $I_{\max} = 8$ and daily demand $D = 2$ boxes.

- **State**: $(I, s)$ where $I \in \{0, \dots, 8\}$ and $s \in \{\text{Normal}, \text{Congestion}, \text{Crisis}\}$.
- **Arrival Success along Chokepoints**: $100\%$ in Normal, $50\%$ in Congestion, $25\%$ in Crisis.
- **Known Policy Parameter**: Import tariff surcharge $\tau$ (USD/box) applies to every ordered box in all states.

### Immediate Cost & Bellman Optimality Equation
Let $\text{arrived} = \text{round}(a \cdot \text{rate}(s))$, $\text{sold} = \min(2, I + \text{arrived})$, $\text{short} = 2 - \text{sold}$, and $I' = \min(8, I + \text{arrived} - \text{sold})$:
$$Q(I, s, a) = (c + \tau)\,a + c_h\,I' + c_{so}\,\text{short} + \gamma \sum_{s'=0}^{2} P(s' \mid s)\,V(I', s')$$
where $c = \$4$, $c_h = \$1$, $c_{so} = \$30$, and $\gamma = 0.90$. Value Iteration updates $V(I, s) \leftarrow \min_a Q(I, s, a)$ until convergence ($\Delta < 10^{-3}$).

### Urgency Score Passed to A\*
$$U(I) = \begin{cases} 1.0 & I = 0 \\ 0.7 & I = 1 \\ 0.4 & I = 2 \\ 0.0 & I \ge 3 \end{cases}$$

---

## 3. Algorithm 3: Dynamic A\* Search
**Course Topic: Classical Search & Heuristics**

### Multimodal Network (7 Nodes, 5 Routes: Jebel Ali → Hamburg)
1. **Sea via Hormuz + Suez**: $\$110$ base, $18$ days *(risk: Hormuz $r=3.0$, Red Sea $r=2.0$)*
2. **Sea via Hormuz + Cape**: $\$185$ base, $31$ days *(risk: Hormuz $r=3.0$)*
3. **Truck to Salalah + Sea via Suez**: $\$140$ base, $18$ days *(bypasses Hormuz; Red Sea $r=2.0$)*
4. **Truck to Salalah + Sea via Cape**: $\$210$ base, $30$ days *(bypasses Hormuz & Suez, $r=0$)*
5. **Air Cargo from Dubai**: $\$300$ base, $1.5$ days *(zero chokepoint risk, $r=0$)*

### Dynamic Edge Cost Formula
$$\text{cost}(e) = b + b \cdot \kappa \cdot p \cdot r + \beta \cdot U \cdot d + \theta \cdot \tau \cdot d$$
with disruption sensitivity $\kappa = 1.5$, time urgency penalty $\beta = \$12\text{/day}$, and tariff-timing exposure $\theta = 0.4\text{ per USD/day}$ ($\approx 8\text{-box shipment} \times 5\%\text{/day}$ chance a pending tariff hike hits before customs clearance — the real-world "front-loading" effect that shifts cargo toward faster corridors or air freight when tariffs rise).

### Provably Admissible & Consistent Heuristic
$$h(n) = c_{\min} \cdot \text{haversine\_km}(n, \text{Hamburg}), \quad c_{\min} = \min_{e \in E} \frac{b(e)}{\text{haversine\_km}(e.u, e.v)}$$
Because all penalties are non-negative ($\text{cost}(u,v) \ge b(u,v) \ge c_{\min} \cdot \text{dist}(u,v)$ for $p, U, \tau \ge 0$) and great-circle distance satisfies the spherical triangle inequality, $h(u) \le \text{cost}(u,v) + h(v)$ (consistent) and $h(n) \le h^*(n)$ (admissible) for every node and every $(p, U, \tau)$.
