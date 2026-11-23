# Supply Chain Vulnerability & Tariff Disruption Advisory System
**AI Fundamentals University Course Final Project**

An end-to-end, 100% offline decision-support application for **Jebel Ali (Dubai) → Hamburg** logistics that integrates three core AI algorithm families:
1. **Hidden Markov Model (HMM)** — *Probabilistic Reasoning / Temporal Models*: Infers latent corridor risk states *(Normal, Congestion, Crisis)* from noisy delay/rate observations via Forward filtering.
2. **Markov Decision Process (MDP)** — *Sequential Decision-Making under Uncertainty*: Computes cost-optimal warehouse replenishment policies via Bellman Value Iteration under known import tariffs and state-dependent delivery rates.
3. **Dynamic A\* Search** — *Classical Search & Heuristics*: Plans cost-minimal multimodal routes across a 7-node network *(Strait of Hormuz, Salalah Overland Bypass, Suez Canal, Cape of Good Hope, Dubai Air Cargo)* using a provably admissible great-circle heuristic.

---

## 📁 Repository Structure
```
supply_chain_advisory/
├── app.py                        # Interactive Streamlit dashboard (offline)
├── requirements.txt              # Python dependency manifest
├── .streamlit/
│   └── config.toml               # Light theme configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Scenario generator & UI presets
│   ├── hmm_module.py             # Forward filtering algorithm & scalar risk mapping
│   ├── mdp_planner.py            # Bellman Value Iteration warehouse optimizer
│   ├── graph_search.py           # 7-node Jebel Ali -> Hamburg network & admissible A*
│   ├── simulation_runner.py      # Closed-loop AI vs baseline simulator & CLI runner
│   ├── map_view.py               # Offline SVG map with animated trailing routes
│   └── ui_components.py          # Interactive formula explorer with variable highlighting
├── tests/
│   └── test_pipeline.py          # 6-suite automated verification tests
└── docs/
    ├── MATH_DOCUMENTATION.md     # Mathematical equations and proofs
    ├── STUDENT_DEFENSE_GUIDE.md  # Oral defense pitch, Q&A, and live demo script
    └── EVALUATION_REPORT.md      # Experimental report (to be updated)
```

---

## 🚀 Quickstart Guide

### Activate Virtual Environment
```bash
cd "/Users/derlyight/Documents/antgrav files/supply_chain_advisory"
source .venv/bin/activate
```

### 1. Run Verification Tests
```bash
python tests/test_pipeline.py
```
Validates HMM probability normalization & purity, MDP Bellman Q-value identities, A\* rerouting across all 5 regimes, tariff-timing sensitivity, heuristic admissibility ($h(n) \le h^*(n)$), and ground-truth closed-loop simulation.

### 2. Run Headless Terminal Simulation
```bash
python src/simulation_runner.py
```
Prints the solved MDP policy table, A\* route selection across the 3 scenario presets, and 15-day AI-vs-Baseline benchmarks.

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to explore:
- **Route Map** with preset buttons (*All Clear*, *Tariff Surge*, *Hormuz Blockage*) and animated cargo trajectories
- **Calculations Tab** with hover-highlighted interactive formulas showing every live number
- **Order Policy (MDP)**, **Simulation & Results**, and **How it works** pipeline diagram
