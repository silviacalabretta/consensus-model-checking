# Consensus Model Checking

Analysis of consensus formation and robustness in finite populations of agents with disruptive individuals.

The project studies two consensus mechanisms:

* the Voter Model;
* the Cross-Inhibition Model;

under two types of disruptive agents:

* zealots, whose opinion is fixed;
* contrarians, whose internal state changes dynamically.

The finite population models are represented as Continuous-Time Markov Chains (CTMCs) in PRISM and analysed using Stormpy. A Python implementation of the Gillespie algorithm is also provided to generate stochastic trajectories of the same CTMC models.

## Current functionality

The repository currently supports:

* four parametric PRISM CTMC models;
* exact probabilistic model checking with Stormpy;
* bounded consensus reachability properties;
* probabilistic consensus-stability properties;
* expected consensus-reaching time through rewards;
* stochastic trajectory generation using the Gillespie algorithm;
* trajectory statistics and visualization;
* single-trajectory plots with majority-label shading;
* mean and variance plots over multiple trajectories.

Automated parameter sweeps and Statistical Model Checking are planned but not yet implemented.

## Models

The following models are available:

| Model identifier       | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `voter_zealots`        | Voter Model with static zealots                 |
| `voter_contrarians`    | Voter Model with dynamic contrarians            |
| `crossinh_zealots`     | Cross-Inhibition Model with static zealots      |
| `crossinh_contrarians` | Cross-Inhibition Model with dynamic contrarians |

The PRISM files are stored in `models/`.

## Repository structure

```text
consensus-model-checking/
├── models/                  # PRISM CTMC models
├── properties/              # Probabilistic and reward properties
├── tests/                   # Automated model and invariant tests
├── util/
│   ├── cli_args.py          # Shared command-line arguments
│   ├── model_builder.py     # PRISM parsing and Storm model construction
│   ├── plot_utils.py        # Trajectory plotting functions
│   └── simulation_types.py  # Trajectory result data classes
├── model_check.py           # Exact probabilistic model checking
├── simulate.py              # Gillespie simulation
├── requirements.txt         # Runtime dependencies
```

## Parameters

The main model parameters are:

| Parameter | Meaning                        |
| --------- | ------------------------------ |
| `N`       | Total population size          |
| `Za`      | Number of A-zealots            |
| `Zb`      | Number of B-zealots            |
| `C`       | Total number of contrarians    |
| `t`       | Consensus-reaching time bound  |
| `h`       | Consensus-stability time bound |

The zealot parameters are used only by the zealot models, while `C` is used only by the contrarian models.

## Environment setup

The project is intended to be run in a Linux or WSL environment with Python and Stormpy available.

Clone the repository and enter its directory:

```bash
git clone https://github.com/silviacalabretta/consensus-model-checking.git
cd consensus-model-checking
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the runtime dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exact model checking

Run exact probabilistic model checking for one model configuration:

```bash
python model_check.py --model voter_zealots --N 20 --Za 2 --Zb 2 --t 35 --h 40
```

Example for a model with contrarians:

```bash
python model_check.py --model crossinh_contrarians --N 20 --C 4 --t 35 --h 40
```

The script builds the complete reachable CTMC and reports:

* number of reachable states;
* number of transitions;
* model-construction time;
* total property-checking time;
* value and checking time of each property.

The properties are defined in:

```text
properties/consensus_prob.props
properties/consensus_time.props
```

## Stochastic simulation

Generate Gillespie trajectories with:

```bash
python simulate.py --model voter_zealots --N 20 --Za 2 --Zb 2 --episodes 100 --max-time 100 --seed 0
```

Example with contrarians:

```bash
python simulate.py --model voter_contrarians --N 20 --C 4 --episodes 100 --max-time 100 --seed 0
```

The simulator reports:

* sample trajectory states;
* fraction of time spent in consensus;
* number of entries into consensus;
* final majority statistics;
* total simulation time.

Complete trajectories are currently stored in memory as `EpisodeResult` objects. Plotting functions are available in `util/plot_utils.py`.

The simulator currently generates and summarizes stochastic trajectories. Formal trajectory monitoring and statistical confidence guarantees will be added as part of the future Statistical Model Checking implementation.

## Running tests

Run the complete test suite from the repository root:

```bash
pytest -q
```

The tests check basic implementation properties such as:

* successful construction of all four CTMCs;
* existence of a unique initial state;
* conservation of the total population;
* conservation of the number of contrarians;
* mutual exclusion of the `maj_a` and `maj_b` labels;
* non-negative transition rates;
* rejection of invalid parameters.

## Development status

Completed:

* definition of the four finite CTMC models;
* model construction with Stormpy;
* exact verification of individual configurations;
* Gillespie trajectory simulation;
* trajectory statistics and plotting.

Planned:

1. automated exact-verification parameter sweeps;
2. CSV storage of experimental results;
3. consensus-probability and state-space-growth plots;
4. validation of simulations against exact model-checking results;
5. Statistical Model Checking with confidence intervals;
6. large-population finite-size experiments.

## Project goal

The main objective is to quantify how zealots and contrarians affect:

* the probability of reaching consensus;
* the probability of maintaining consensus;
* the expected time required to reach consensus;
* the computational scalability of exact verification;
* the behaviour of larger populations through stochastic simulation.


## References

[1] J. Klein and T. Petrov, "Quantifying consensus in stochastic swarms with disruptive individuals," 2025 European Control Conference (ECC), Thessaloniki, Greece, 2025, pp. 272-277, doi: 10.23919/ECC65951.2025.11186823.