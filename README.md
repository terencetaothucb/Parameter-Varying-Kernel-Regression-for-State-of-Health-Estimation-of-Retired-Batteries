# CDC2026 Repository Guide

This repository is organized around the current CDC paper:

`Parameter-Varying Observer Synthesis for Retired Battery Health Estimation under Pulse Testing at Random States of Charge`

## Directory layout

- `Data/raw/20Ah_LFP/`
  Original 20Ah pulse-test Excel files used in the paper.
- `Data/processed/data4model_20ah.xlsx`
  Processed 20Ah-only dataset used by all experiments and figures.
- `scripts/prepare_20ah_dataset.py`
  Rebuilds `Data/processed/data4model_20ah.xlsx` from the raw 20Ah Excel files.
- `scripts/observer_benchmark_utils.py`
  Shared data-loading, baseline, and helper utilities.
- `scripts/run_parameter_varying_observer_benchmarks.py`
  Runs the main Case 1/2/3 unseen-SOC interpolation benchmarks and writes `results/benchmark_results_20ah.xlsx`.
- `scripts/plot_fixed_soc_conditional_observability.py`
  Generates Fig. 2.
- `scripts/plot_unseen_soc_interpolation_performance.py`
  Generates Fig. 3.
- `scripts/plot_pulse_width_impact.py`
  Generates the pulse-width robustness figure and its detail CSV.
- `scripts/plot_pulse_amplitude_impact.py`
  Generates the pulse-amplitude robustness figure and its detail CSV.
- `results/`
  Benchmark workbook and figure-level detail CSV files.
- `figures/`
  Paper-ready PNG and PDF figures.

## Main files for the paper

- `results/benchmark_results_20ah.xlsx`
- `figures/fixed_soc_conditional_observability.pdf`
- `figures/unseen_soc_interpolation_performance.pdf`
- `figures/pulse_width_impact.pdf`
- `figures/pulse_amplitude_impact.pdf`

## Reproduction order

1. `python3 scripts/prepare_20ah_dataset.py`
2. `python3 scripts/run_parameter_varying_observer_benchmarks.py`
3. `python3 scripts/plot_fixed_soc_conditional_observability.py`
4. `python3 scripts/plot_unseen_soc_interpolation_performance.py`
5. `python3 scripts/plot_pulse_width_impact.py`
6. `python3 scripts/plot_pulse_amplitude_impact.py`
