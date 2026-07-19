# Consensus Model Checking

Analysis of consensus formation and robustness in finite populations of agents with disruptive individuals, using Continuous-Time Markov Chain (CTMC) model checking and Gillespie simulation.

## Project Structure

```
consensus-new/
├── model_check.py              # Exact model checking (single configuration)
├── simulate.py                 # Gillespie simulation (single configuration)
├── src/
│   ├── checking.py             # Exact model checking via stormpy
│   ├── cli.py                  # CLI argument parsers
│   ├── estimation.py           # Probability estimation and Wilson CI
│   ├── gillespie_py.py         # Gillespie algorithm for CTMC simulation, python based
│   ├── gillespie_storm.py      # Gillespie algorithm for CTMC simulation, storm based
│   ├── model.py                # CTMC building and helpers
│   ├── monitors.py             # Trajectory monitors (reachability, holding)
│   ├── plotting.py             # Shared plotting utilities
│   └── types.py                # All dataclasses
├── experiments/
│   ├── validate.py             # Validate simulation against exact results
│   ├── exact_sweep.py          # Exact model checking over parameter grid
│   ├── smc.py                  # Statistical Model Checking (stable consensus)
├── experiments/
│   ├── exact_sweep.py              # Exact model-checking parameter sweep
│   ├── grid_common.py              # Shared utilities for parameter sweeps
│   ├── smc.py                      # SMC for a single configuration
│   ├── smc_sweep.py                # Large-population SMC parameter sweep
│   ├── smc_h_sweep.py              # SMC sweep over holding times
│   └── validate_simulators.py      # Common simulator and monitors validation
├── visualization/              # Sweep plotting scripts
├── models/                     # PRISM model files
├── properties/                 # PRISM property files
├── results/                    # Generated CSV files
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

### Gillespie simulation (stormpy based)
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

## Results

The experiments compare the robustness of the four models for options with qualities \(q_A = 1.05\) and \(q_B = 0.95\), using exact model checking for smaller populations and Statistical Model Checking with 4500 simulations per configuration for populations up to \(N=4000\).

The main findings are:

- cross-inhibition maintains stable consensus under substantially larger disruptor fractions than the voter model;
- contrarians generally cause an earlier loss of stable consensus than zealots;
- in cross-inhibition, a moderate fraction of zealots can increase the probability of stable consensus on the higher-quality option;
- increasing the population size makes the transition between stable and unstable regimes progressively sharper;
- majority reachability may remain high even when stable consensus is unlikely, showing that disruptors first reduce consensus persistence before preventing majority formation;
- the direct Python simulator scales approximately linearly with the population size and enables experiments beyond the range of exact state-space construction.

Approximate large-population transition regions for stable \(A\)-consensus are:

| Model | Disruptor-fraction transition |
| --- | ---: |
| Voter + zealots | 0.25–0.32 |
| Voter + contrarians | 0.12–0.15 |
| Cross-inhibition + zealots | 0.45–0.50 |
| Cross-inhibition + contrarians | 0.24–0.28 |

The complete model definitions, deterministic analysis, experimental methodology, and results are available in the [project report](project_report.pdf).

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
[2] Luca Bortolussi, Jane Hillston, Diego Latella, Mieke Massink, Continuous approximation of collective system behaviour: A tutorial, Performance Evaluation, Volume 70, Issue 5, 2013, Pages 317-349, ISSN 0166-5316, https://doi.org/10.1016/j.peva.2013.01.001.
