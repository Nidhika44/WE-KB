# WE-KB: Training-Free, Knowledge-Seeded, Radio-Aware Metaheuristic Offloading for Multi-Access Edge Computing

Official implementation of **WE-KB-FPA** (Wireless-Edge Knowledge Base Flower Pollination Algorithm) for multi-user, multi-edge computation offloading under dynamic fading channels.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 111 Passed](https://img.shields.io/badge/tests-111%20passed-brightgreen.svg)](tests/)

---

## Author & Contact

- **Dr. Nidhika Chauhan** — [`nidhi29.chauhan@gmail.com`](mailto:nidhi29.chauhan@gmail.com)

---

## Overview

Joint task offloading, CPU allocation, and bandwidth allocation in Multi-Access Edge Computing (MEC) is a non-convex Mixed-Integer Nonlinear Program (MINLP). Exact solvers scale poorly with network size, deep reinforcement learning (DRL) requires substantial offline training and is sensitive to distribution shift under channel variations, and conventional metaheuristics search with radio-blind operators where channel conditions enter only through fitness evaluation.

**WE-KB-FPA** is a *training-free, memory-augmented, radio-aware optimizer* tailored for dynamic wireless environments where channel statistics vary or training data are scarce:
1. **Knowledge-Base Seeding:** Historical high-quality allocation states are stored and indexed; top-$k$ nearest solutions seed the initial metaheuristic population to accelerate convergence.
2. **Physics-Guided Deadline Pruning:** Analytical lower bounds filter out execution targets that cannot physically satisfy individual task deadlines over degraded radio links, reducing the combinatorial search space prior to exploration.
3. **Channel-Adaptive Search Operators:** The Lévy flight step size and global/local switching probability $p_s$ dynamically adapt to instantaneous SINR and link spectral efficiency variability.

---

## Project Structure

```text
we_kb_fpa/
├── we_kb_fpa/                   # Core simulation package
│   ├── __init__.py
│   ├── we_kb_fpa.py            # WE-KB-FPA optimizer & Lévy distribution sampler
│   ├── config.py               # Network, Task, Channel, Cost, and Optimizer configs
│   ├── task_model.py           # Task generator and Poisson arrival model
│   ├── channel_model.py        # Rayleigh/Rician fading, path loss, shadowing, SINR
│   ├── environment_model.py    # Multi-user, edge, cloud nodes & Gauss-Markov mobility
│   ├── cost_engine.py          # Delay, energy, deadline & resource violation evaluator
│   ├── knowledge_base.py       # Experience memory, KNN retrieval, and candidate pruning
│   ├── allocation.py           # CPU and uplink bandwidth allocation decoders
│   ├── anytime.py              # Anytime computation control and deadline budgeting
│   ├── baselines.py            # PSO, GA, Standard FPA, Random, Greedy baselines
│   ├── exact_solver.py         # Exact MINLP solver using Gurobi / SciPy benchmarks
│   ├── dqn.py                  # DRL / Deep Q-Network baseline controller
│   ├── droo.py                 # Deep Reinforcement Learning Offloading benchmark
│   ├── simulation_executor.py  # Multi-slot dynamic simulation runner
│   ├── checkpoint.py           # Atomic state serialization and reproducible hashing
│   └── publication_plots.py    # Publication-ready matplotlib figure stylers
├── experiments/                # Reproducibility benchmarks & CLI drivers
│   ├── __init__.py
│   ├── run_all.py              # Master CLI experiment runner
│   ├── plot_results.py         # Generation of publication figures
│   ├── run_scalability.py      # User scaling sweep (N = 10 to 500 users)
│   ├── run_channel_degradation.py # Channel SINR degradation sweep (-5 dB to 20 dB)
│   ├── run_workload.py         # Task size and workload variation sweep
│   ├── run_ablation.py         # 16-configuration factorial ablation study
│   ├── run_optimality_gap.py   # Comparison against exact MINLP oracle
│   ├── run_topology_shock.py   # Exogenous link shock & dynamic adaptation
│   ├── run_joint.py            # Joint optimization experiments
│   └── research_report.py      # Statistical summaries, bootstrap CI95 computation
├── tests/                      # Unit & regression test suite (111 tests)
├── pyproject.toml              # Build system specification
├── requirements.txt            # Pinned reproducible dependencies
└── LICENSE                     # MIT License
```

---

## Installation & Setup

### Prerequisites

- Python 3.12 or newer
- Virtual environment tool (`venv`)

### 1. Clone the repository

```bash
git clone https://github.com/Nidhika44/WE-KB.git
cd WE-KB
```

### 2. Create and activate a virtual environment

```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

*Optional packages:*
```bash
# For Gurobi exact solver benchmark:
pip install gurobipy

# For Parquet storage:
pip install pyarrow
```

---

## Verifying the Installation

Run the complete test suite (111 unit and regression tests) to verify environment integrity:

```bash
python -m unittest discover -s tests -v
```

All 111 tests verify:
- Numerical correctness of channel fading models and path loss equations
- Exact cost calculations and penalty functions
- Deadline pruning validity
- Knowledge base storage, retrieval, and seeding fidelity
- Optimizer convergence and baseline implementations
- Statistical metrics collector and reproducibility checksums

---

## Running Experiments

The framework provides a command-line interface `we-kb-fpa` (or via Python module `python -m experiments.run_all`):

### Quick Validation Run

Run all experimental benchmarks with reduced trial counts to verify the pipeline:

```bash
python -m experiments.run_all --quick
```

### Full Benchmark Suite

Run the full configured evaluation suite:

```bash
python -m experiments.run_all --full
```

### Rigorous Evaluation (30 Trials, 30 Seeds)

```bash
python -m experiments.run_all --rigorous
```

### Individual Experiment Modules

- **Scalability Analysis ($N \in \{10, 20, 50, 100, 200\}$ users):**
  ```bash
  python -m experiments.run_all --experiment scalability
  ```

- **Channel Degradation ($\text{SINR} \in \{-5, 0, 5, 10, 15, 20\}\text{ dB}$):**
  ```bash
  python -m experiments.run_all --experiment channel
  ```

- **Workload Variation ($D_k \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0\}\text{ Mbits}$):**
  ```bash
  python -m experiments.run_all --experiment workload
  ```

- **Factorial Component Ablation (16 Combinations):**
  ```bash
  python -m experiments.run_all --experiment ablation
  ```

- **Optimality Gap (vs Exact MINLP Solver):**
  ```bash
  python -m experiments.run_all --experiment optimality
  ```

- **Topology Shock & Dynamic Adaptation:**
  ```bash
  python -m experiments.run_all --experiment shock
  ```

### Plotting Figures

Generate plots from experiment outputs:

```bash
python -m experiments.plot_results
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
