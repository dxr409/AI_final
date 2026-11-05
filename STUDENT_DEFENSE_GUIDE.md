# Student Oral Defense & Presentation Guide
## AI Fundamentals Course Project

Use this guide to prepare for your presentation and oral exam with your teacher. Everything is explained in plain English with simple analogies and word-for-word answers you can say out loud.

---

## 1. The 30-Second Elevator Pitch
> *"Professor, our project solves a realistic problem: imagine you run a company in Europe that imports laptops from Asia.*  
> *When shipping canals get blocked or governments impose tariffs, you face three challenges:*  
> *1. You can't see the disruption directly, so you must **infer the hidden risk** from noisy delay and price reports — we use a **Hidden Markov Model (HMM)**.*  
> *2. You must decide **how many boxes to order** each day so your warehouse doesn't run out of stock — we use a **Markov Decision Process (MDP)** solved with Value Iteration.*  
> *3. You must **dynamically route the cargo** (Ship, Train, or Airplane) from Shanghai to Hamburg — we use **A\* Search** where edge costs dynamically update based on the HMM risk and the MDP warehouse urgency.*  
> *Instead of separate toy scripts, the output of one algorithm directly feeds into the next."*

---

## 2. The 3 Algorithms Explained in Plain English

### Algorithm 1: Hidden Markov Model (HMM)
- **Course Topic**: Probabilistic Reasoning / Temporal Models
- **In simple words**: It's a "risk detector".
- **The intuition**: We can't see what's happening at sea. We only see reports: "Cheap & On-Time", "Moderate Delay", or "Expensive & Stuck". The HMM keeps a probability belief vector `[P(Normal), P(Congestion), P(Crisis)]` and updates it every day using the **Forward Algorithm**.
- **How to explain the math**:  
  *"It's just two steps: first, multiply yesterday's belief by the Transition Matrix A to predict today's state. Second, multiply by the Emission Matrix B for what we observed, and divide by the sum so the percentages add up to 100%."*

### Algorithm 2: Markov Decision Process (MDP)
- **Course Topic**: Sequential Decision-Making under Uncertainty
- **In simple words**: It's an "order quantity planner".
- **The intuition**: You have a warehouse with 0 to 8 boxes. Customers buy 2 boxes each day. If your warehouse is empty, you pay a steep \$30 stockout penalty. If you order too much, you pay \$1 storage fee. If a crisis happens, shipments might get stuck at customs.
- **How it solves it**:  
  *"We use Bellman Value Iteration. For every warehouse level (0 to 8 boxes) and every risk state (Normal, Congestion, Crisis), we find the order quantity (0, 2, 4, or 6 boxes) that minimizes: Today's Cost + (0.90 × Tomorrow's Expected Future Cost). The resulting table tells the manager exactly what to do."*

### Algorithm 3: Dynamic A* Search
- **Course Topic**: Classical Search & Heuristics
- **In simple words**: It's a "smart GPS delivery router".
- **The choices**:
  - **Suez Canal (Sea)**: Cheapest (\$100), but takes 20 days and passes through the vulnerable Suez Canal.
  - **Cape Route (Sea)**: Safe detour around Africa (\$160), but takes 30 days.
  - **Trans-Asia Rail**: Train across Asia (\$200), fast (10 days), bypasses sea canals completely.
  - **Air Cargo**: Airplane (\$400), ultra-fast (2 days).
- **How it connects to HMM & MDP**:  
  $$\text{Edge Cost} = \text{Base Cost} + (\text{HMM Disruption Risk} \times \text{Risk Factor}) + (\text{MDP Urgency Score} \times 12 \times \text{Days})$$
  - If HMM detects that Suez is blocked $\to$ Suez becomes expensive $\to$ A\* switches to the **Cape Sea Detour** or **Rail**.
  - If MDP detects the warehouse has 0 stock (high urgency) $\to$ Slow 20-30 day ships get heavily penalized $\to$ A\* switches to **Express Train** or **Air Cargo**!

---

## 3. How the 3 Algorithms Talk to Each Other (The Pipeline)

```
[Daily Operational Report: "Expensive & Stuck"]
                      │
                      ▼
        ┌───────────────────────────┐
        │  1. HMM (hmm_module.py)   │
        │  Calculates Disruption    │
        │  Risk Probability: 85%    │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  2. MDP (mdp_planner.py)  │   │ 3. A* (graph_search.py)   │
│  State: (0 boxes, Crisis) │   │ Dynamic Edge Costs:       │
│  Action: Order 0 boxes    │   │ • Suez penalized by HMM   │
│  Urgency Score: 100%      │──▶│ • Days penalized by MDP   │
└───────────────────────────┘   └─────────────┬─────────────┘
                                              │
                                              ▼
                                 [A* Selects: EXPRESS TRAIN]
                                 Avoids blocked canal & arrives
                                 in 10 days instead of 30!
```

---

## 4. Top 5 Questions Your Teacher Will Ask (And What to Answer)

### Q1: *"Why did you use an HMM instead of just checking if the delay was high today?"*
> **Answer**: *"Because a single high delay reading could just be noisy random luck (like a minor rainstorm). An HMM remembers past history through the temporal transition matrix $A$. It only transitions to the 'Crisis' state if delays stay consistently high over multiple days, preventing our supply chain from overreacting to single noisy outliers."*

### Q2: *"In the MDP, why Value Iteration instead of a simple rule like 'always order 2 boxes'?"*
> **Answer**: *"A simple rule cannot adapt to shipping disruptions. When ports are congested and delivery success drops from 100% to 50%, our MDP automatically increases order quantities to build a safety buffer. But when a tariff crisis hits with extra surcharges, our MDP automatically cuts orders to avoid expensive tariff penalties. Value Iteration mathematically guarantees the cost-minimal policy under these uncertain transition probabilities."*

### Q3: *"What is the state space of your MDP?"*
> **Answer**: *"The state is a pair: `(Current Warehouse Inventory, Risk State)`. Inventory ranges from 0 to 8 boxes, and the Risk State is one of 3 states from the HMM: Normal, Congestion, or Crisis. That gives $9 \times 3 = 27$ total states. For each state, the algorithm chooses an action from `[0, 2, 4, 6]` boxes."*

### Q4: *"In A*, what is your heuristic function $h(n)$ and is it admissible?"*
> **Answer**: *"Yes, professor. The heuristic $h(n)$ is the straight-line Euclidean distance from the current node to Hamburg, multiplied by a scaling factor of 8.0. Because a straight line is the shortest possible path in Euclidean geometry, $h(n)$ will never overestimate the true remaining cost ($h(n) \le h^*(n)$). This makes it admissible, which guarantees that A\* will find the true optimal route."*

### Q5: *"Can you show me where the three algorithms are actually connected in code?"*
> **Answer**: *"Yes, in `simulation_runner.py` inside the main loop:
> - Line 66: `risk_prob = hmm.get_disruption_probability()` (HMM gives risk)
> - Line 70: `order_qty = policy[current_inv, state]` and `urgency = mdp.get_urgency_score(current_inv)` (MDP gives urgency)
> - Line 74: `network.a_star_search(..., disruption_prob=risk_prob, urgency_score=urgency)` (A\* takes both inputs as parameters!)"*

---

## 5. Live Demonstration Script for Class

### Option A: Running from the Terminal (Clean & Fast)
```bash
python3 simulation_runner.py
```
1. Point to **Step 1**: Show the printed MDP policy table. Explain: *"Look at row 0 (empty warehouse): in Normal state it orders 2 boxes, in Congestion it orders 4 boxes, and in Crisis it orders 0."*
2. Point to **Step 2**: Show the 3 scenario comparisons. Explain: *"Look how A\* picks Suez Canal under zero risk, shifts to Cape of Good Hope when Suez is blocked, and shifts to Train when the warehouse is empty!"*
3. Point to **Step 3**: Walk through the 12-day simulation table.

### Option B: Running the Streamlit Web Dashboard (Visual & Interactive)
```bash
./.venv/bin/streamlit run app.py
```
1. Open `http://localhost:8501`.
2. Go to **Tab 1 ("Interactive A\* Map & Rerouting")**.
3. Move the **Corridor Disruption Risk** slider from 0% to 90%:
   - Watch the red line on the map switch from **Suez Canal (Blue)** to **Cape of Good Hope (Detour)**.
4. Move the **Warehouse Stockout Urgency** slider up to 100%:
   - Watch the route switch to **Express Train (Orange)** or **Air Cargo (Green)**.
5. Go to **Tab 2** to show the MDP policy table and explain why the numbers make business sense.

