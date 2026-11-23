# Student Oral Defense & Presentation Guide
## AI Fundamentals Course Project

Use this guide to prepare for your presentation and oral defense.

---

## 1. The 30-Second Elevator Pitch
> *"Professor, our project solves a realistic logistics crisis: imagine a European electronics importer shipping laptops from **Jebel Ali (Dubai)** in the Persian Gulf to **Hamburg, Germany**.*  
> *When the **Strait of Hormuz** faces disruption risk or governments raise import tariffs, we face three tightly coupled challenges:*  
> *1. We cannot observe the true corridor state directly, so a **Hidden Markov Model (HMM)** filters noisy daily rate and delay signals into a disruption probability $p$.*  
> *2. An **MDP** solved via Bellman Value Iteration decides how many boxes to reorder under known tariff surcharges and calculates warehouse stockout urgency $U$.*  
> *3. **Dynamic A\* Search** picks the optimal multimodal route—sailing via Hormuz & Suez, trucking overland around Hormuz to Salalah (Oman), diverting around the Cape of Good Hope, or escalating to Dubai Air Cargo—using edge costs driven directly by HMM risk $p$ and MDP urgency $U$."*

---

## 2. The 3 Algorithms in Plain English

### Algorithm 1: Hidden Markov Model (HMM)
- **Course Topic**: Probabilistic Reasoning / Temporal Models
- **Role**: Corridor risk estimator (`hmm_module.py`).
- **How it works**: Updates `[P(Normal), P(Congestion), P(Crisis)]` every day via Forward filtering (`predicted = b_{t-1} @ A`, multiply by emission likelihood `B[:, obs]`, normalize by `Z`). Outputs scalar risk `p = 0.4·P(Congestion) + 1.0·P(Crisis)`.

### Algorithm 2: Markov Decision Process (MDP)
- **Course Topic**: Sequential Decision-Making under Uncertainty
- **Role**: Optimal inventory replenishment (`mdp_planner.py`).
- **How it works**: Solves Bellman Value Iteration over `9 × 3 = 27` `(Inventory, RiskState)` states with actions `{0, 2, 4, 6}` boxes, wholesale cost `$4`, known tariff `τ`, holding fee `$1`, stockout penalty `$30`, and discount `γ = 0.90`. Outputs order quantity and stockout urgency `U` (`1.0` at 0 stock down to `0.0` at ≥3 stock).

### Algorithm 3: Dynamic A\* Search
- **Course Topic**: Classical Search & Heuristics
- **Role**: Multimodal route planner (`graph_search.py`).
- **The 5 Route Choices (Jebel Ali → Hamburg)**:
  1. **Sea via Hormuz + Suez**: `$110` base, `18 days` (passes Hormuz + Red Sea).
  2. **Sea via Hormuz + Cape**: `$185` base, `31 days` (passes Hormuz).
  3. **Truck to Salalah + Sea via Suez**: `$140` base, `18 days` (overland truck bypasses Hormuz).
  4. **Truck to Salalah + Sea via Cape**: `$210` base, `30 days` (bypasses both Hormuz & Suez).
  5. **Air Cargo from Dubai**: `$300` base, `1.5 days` (fastest, zero maritime chokepoint risk).
- **Dynamic Edge Cost**:
  $$\text{cost}(e) = b + b \cdot 1.5 \cdot p \cdot r + 12 \cdot U \cdot d + 0.4 \cdot \tau \cdot d$$

---

## 3. Pipeline Architecture
```
[Daily Observation: "Expensive & Stuck"]
                  │
                  ▼
     ┌─────────────────────────┐
     │ 1. HMM (hmm_module.py)  │
     │ Infers state & risk p   │
     └────────────┬────────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
┌───────────────────┐   ┌───────────────────────────┐
│ 2. MDP            │   │ 3. Dynamic A*             │
│ (mdp_planner.py)  │   │ (graph_search.py)         │
│ Picks order qty & │──▶│ Edge costs use p (HMM)    │
│ urgency U(stock)  │   │ and urgency U (MDP)       │
└───────────────────┘   └─────────────┬─────────────┘
                                      ▼
                     [Selects Optimal Multimodal Route]
```

---

## 4. Top 5 Teacher Questions & Answers

### Q1: *"Why use an HMM instead of a simple threshold on today's delay?"*
> *"A single bad reading could just be random noise. The HMM remembers prior beliefs through transition matrix $A$, so it shifts smoothly into Congestion or Crisis only when signals persist across multiple days."*

### Q2: *"Why solve an MDP with Value Iteration instead of a fixed reorder rule?"*
> *"A fixed rule cannot adapt simultaneously to customs tariffs and chokepoint arrival drops. Bellman Value Iteration automatically balances `$30` stockout penalties against `$1` holding fees and known tariff surcharges across uncertain future transitions."*

### Q3: *"What is your MDP state and action space?"*
> *"Each state is a pair `(Inventory, RiskState)` with `Inventory ∈ {0..8}` and `RiskState ∈ {Normal, Congestion, Crisis}`, giving `27` total states. Actions are `{0, 2, 4, 6}` boxes."*

### Q4: *"What is your A\* heuristic $h(n)$ and why is it admissible?"*
> *"Our heuristic is `h(n) = c_min × haversine_km(n, Hamburg)`, where `c_min` is the minimum base cost per kilometer across all edges in the network. Because disruption and urgency penalties are always non-negative, every edge costs at least `c_min × distance(u, v)`, and by the spherical triangle inequality great-circle distance never overestimates path distance. Therefore `h(n) <= h*(n)` (admissible) and `h(u) <= cost(u, v) + h(v)` (consistent) for every node and every `(p, U)`—which you can also verify live inside the Calculations tab."*

### Q5: *"Where are the three algorithms actually connected in code?"*
> *"In `simulation_runner.py` inside `run_closed_loop`:*
> - *Lines 68–70: `hmm.update_belief(obs)` computes `p_risk` and `inferred_s`.*
> - *Lines 73–75: `mdp.get_urgency_score(stock)` computes `urgency` and `policy[stock, inferred_s]` selects `order`.*
> - *Lines 76–82: `network.a_star_search(..., disruption_prob=p_risk, urgency_score=urgency, tariff_per_box=tariff)` feeds all inputs directly into A\*."*

### Q6: *"How does the import tariff affect routing in A\*?"*
> *"When a tariff surge looms, every extra day at sea increases the chance that goods clear customs after the new tariff takes effect (known as front-loading). We model that expected tariff-timing exposure as `0.4 × τ × transit_days`. Under Hormuz Blockage, raising the tariff from `$2` to `$8` causes the 30-day Cape detour's tariff exposure (`$96`) to outweigh Air Cargo's base premium, flipping A\* cleanly to Dubai Air Cargo."*

---

## 5. Live Demonstration Script

### Terminal Demo
```bash
python tests/test_pipeline.py
python src/simulation_runner.py
```

### Interactive Dashboard Demo (`streamlit run app.py`)
1. Open `http://localhost:8501` on **Tab 1 (Route Map)**.
2. Click the **3 Quick Scenario buttons** one by one:
   - **All Clear** (`5%` risk, stock `5`, tariff `$2`) → **Sea via Hormuz + Suez** (`18 days`).
   - **Tariff Surge** (`40%` risk, stock `3`, tariff `$12`) → **Truck to Salalah + Sea via Suez** (bypasses Hormuz).
   - **Hormuz Blockage** (`90%` risk, stock `3`, tariff `$2`) → **Truck to Salalah + Sea via Cape** (bypasses both Hormuz & Suez).
3. On **Hormuz Blockage**, drag the **Import Tariff** slider from `2` to `8`: the route flips from **Cape** to **Air Cargo from Dubai**! (Similarly, dragging **Warehouse Stock** down to `0` also escalates to **Air Cargo**.)
4. Open **Tab 2 (Calculations)** and hover symbols like `p`, `U`, `tau`, or `b` to show the live substituted numbers and explanation cards lighting up together.
5. Open **Tab 4 (Simulation & Results)** to compare **Integrated AI vs Static Baseline** stockouts and costs over 15 days.
