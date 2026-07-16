# Consensus Model Checking

Analysis of consensus formation and robustness in finite populations of agents with disruptive individuals, using Continuous-Time Markov Chain (CTMC) model checking and Gillespie simulation.

## Project Structure

```
consensus-new/
├── model_check.py              # Exact model checking (single configuration)
├── simulate.py                 # Gillespie simulation (single configuration)
├── src/
│   ├── types.py                # All dataclasses (ModelParams, etc.)
│   ├── cli.py                  # CLI argument parsers
│   ├── model.py                # CTMC building and helpers
│   ├── checking.py             # Exact model checking via stormpy
│   ├── gillespie.py            # Gillespie algorithm for CTMC simulation
│   ├── monitors.py             # Trajectory monitors (reachability, holding)
│   ├── estimation.py           # Probability estimation and Wilson CI
│   └── plotting.py             # Shared plotting utilities
├── experiments/
│   ├── validate.py             # Validate simulation against exact results
│   ├── exact_sweep.py          # Exact model checking over parameter grid
│   ├── smc.py                  # Statistical Model Checking (stable consensus)
│   └── plot_model_check_results.py  # Plot grid results from exact_sweep.py
├── models/                     # PRISM model files
├── properties/                 # PRISM property files
└── tests/
    ├── test_model_building.py  # Model construction tests
    └── test_conservation.py    # Population conservation tests
```

## Quick Start

### Exact model checking (single configuration)
```bash
python model_check.py --model voter_zealots --N 20 --Z 4 --t 35 --h 40
```

You can also set `--qa` and `--qb` to tune interaction rates, and specify `--Za`/`--Zb` individually instead of `--Z`:
```bash
python model_check.py --model voter_zealots --N 20 --Za 3 --Zb 1 --qa 1.2 --qb 0.8
```

### Gillespie simulation
```bash
python simulate.py --model voter_zealots --N 20 --Z 4 --episodes 1000 --max-time 100 --seed 42
```

### Validate simulation against exact results
```bash
python experiments/validate.py --model voter_zealots --N 20 --Z 4 --episodes 2000
```

### Exact model checking over parameter grid
```bash
python experiments/exact_sweep.py
```

### Statistical Model Checking
```bash
python experiments/smc.py --model voter_zealots --N 20 --Z 4 --episodes 4500
```

### Plot grid results
```bash
python experiments/plot_model_check_results.py --input results/exact_sweep.csv --output-dir results/plots
```

## Models

| Model | Mechanism | Disruptors |
|-------|-----------|------------|
| `voter_zealots` | Voter rule | Fixed zealots (Za, Zb) |
| `voter_contrarians` | Voter rule | Adaptive contrarians (C) |
| `crossinh_zealots` | Cross-inhibition | Fixed zealots (Za, Zb) |
| `crossinh_contrarians` | Cross-inhibition | Adaptive contrarians (C) |

## Setup

The project requires [stormpy](https://moves-rwth.github.io/stormpy/), so it is suggested to work in a Linux or WSL environment.

Clone the repository and enter its directory:

```bash
git clone https://github.com/silviacalabretta/consensus-model-checking
cd consensus-model-checking
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the project in editable mode (this also installs all dependencies):

```bash
pip install -e .
```

For development tools (pytest), install the dev extras:

```bash
pip install -e ".[dev]"
```


## Tests

```bash
pytest tests/
```


## References

[1] J. Klein and T. Petrov, "Quantifying consensus in stochastic swarms with disruptive individuals," 2025 European Control Conference (ECC), Thessaloniki, Greece, 2025, pp. 272-277, doi: 10.23919/ECC65951.2025.11186823.