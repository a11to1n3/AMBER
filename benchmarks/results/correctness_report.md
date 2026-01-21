# Comprehensive Correctness Benchmark Report

**Generated**: 2026-01-21 01:34:36

## Metric Categories

| Category | Description |
|----------|-------------|
| Conservation | Physical quantities preserved (total wealth, population) |
| Statistical | Output matches known distributions (Boltzmann, diffusion) |
| Reproducibility | Same seed produces identical results |
| Emergent | Expected behaviors emerge (Gini increase, monotonic recovery) |
| Precision | Numerical errors don't accumulate |
| Edge Cases | Boundary conditions handled (single agent, zero steps) |

## Summary by Framework

| Framework | Passed | Total | Score |
|-----------|--------|-------|-------|
| AMBER | 10 | 11 | 91% |
| AgentPy | 10 | 11 | 91% |
| Mesa | 9 | 11 | 82% |

## Detailed Results

### Conservation

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | total_wealth_error | ✅ | Absolute error: 0.00e+00 |
| AMBER | sir_epidemic | population_error | ✅ | Max step error: 0 |
| AgentPy | wealth_transfer | total_wealth_error | ✅ | Absolute error: 0.00e+00 |
| AgentPy | sir_epidemic | population_error | ✅ | Max step error: 0 |
| Mesa | wealth_transfer | total_wealth_error | ✅ | Absolute error: 0.00e+00 |
| Mesa | sir_epidemic | population_error | ✅ | Max step error: 0 |
| Melodie | wealth_transfer | total_wealth_error | ✅ | Absolute error: 0.00e+00 |
| Melodie | sir_epidemic | population_error | ✅ | Max step error: 0 |
| SimPy | wealth_transfer | total_wealth_error | ✅ | Absolute error: 0.00e+00 |
| SimPy | sir_epidemic | population_error | ✅ | Max step error: 0 |

### Statistical

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | variance_growth | ❌ | Var ratio: 0.00x |
| AMBER | random_walk | mean_squared_displacement | ✅ | MSD: 1503.85 |
| AMBER | sir_epidemic | attack_rate | ✅ | Attack rate: 8.0% |
| AgentPy | wealth_transfer | variance_growth | ❌ | Var ratio: 0.01x |
| AgentPy | random_walk | mean_squared_displacement | ✅ | MSD: 1848.45 |
| AgentPy | sir_epidemic | attack_rate | ✅ | Attack rate: 10.0% |
| Mesa | wealth_transfer | variance_growth | ❌ | Var ratio: 0.00x |
| Mesa | random_walk | mean_squared_displacement | ✅ | MSD: 1633.51 |
| Mesa | sir_epidemic | attack_rate | ✅ | Attack rate: 12.0% |
| Melodie | wealth_transfer | variance_growth | ❌ | Var ratio: 0.01x |
| Melodie | random_walk | mean_squared_displacement | ✅ | MSD: 100.00 |
| Melodie | sir_epidemic | attack_rate | ✅ | Attack rate: 0.0% |
| SimPy | wealth_transfer | variance_growth | ❌ | Var ratio: 0.01x |
| SimPy | random_walk | mean_squared_displacement | ✅ | MSD: 100.00 |
| SimPy | sir_epidemic | attack_rate | ✅ | Attack rate: 0.0% |

### Reproducibility

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| AgentPy | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| Mesa | wealth_transfer | seed_determinism | ❌ | Non-deterministic! |
| Melodie | wealth_transfer | seed_determinism | ❌ | Non-deterministic! |
| SimPy | wealth_transfer | seed_determinism | ❌ | Non-deterministic! |

### Emergent

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | gini_increase | ✅ | Gini change: +0.0366 |
| AMBER | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| AgentPy | wealth_transfer | gini_increase | ✅ | Gini change: +0.0420 |
| AgentPy | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| Mesa | wealth_transfer | gini_increase | ✅ | Gini change: +0.0354 |
| Mesa | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| Melodie | wealth_transfer | gini_increase | ❌ | Gini change: +0.0000 |
| Melodie | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| SimPy | wealth_transfer | gini_increase | ❌ | Gini change: +0.0000 |
| SimPy | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |

### Precision

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | error_accumulation | ✅ | Error at 50 steps: 0.00e+00, at 500: 0.00e+00 |
| AgentPy | wealth_transfer | error_accumulation | ✅ | Error at 50 steps: 0.00e+00, at 500: 0.00e+00 |
| Mesa | wealth_transfer | error_accumulation | ✅ | Error at 50 steps: 0.00e+00, at 500: 0.00e+00 |
| Melodie | wealth_transfer | error_accumulation | ✅ | Error at 50 steps: 0.00e+00, at 500: 0.00e+00 |
| SimPy | wealth_transfer | error_accumulation | ✅ | Error at 50 steps: 0.00e+00, at 500: 0.00e+00 |

### Edge Case

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | single_agent | ✅ | Single agent handled |
| AMBER | wealth_transfer | zero_steps | ✅ | Zero steps handled |
| AgentPy | wealth_transfer | single_agent | ✅ | Single agent handled |
| AgentPy | wealth_transfer | zero_steps | ✅ | Zero steps handled |
| Mesa | wealth_transfer | single_agent | ✅ | Single agent handled |
| Mesa | wealth_transfer | zero_steps | ✅ | Zero steps handled |
| Melodie | wealth_transfer | single_agent | ✅ | Single agent handled |
| Melodie | wealth_transfer | zero_steps | ✅ | Zero steps handled |
| SimPy | wealth_transfer | single_agent | ✅ | Single agent handled |
| SimPy | wealth_transfer | zero_steps | ✅ | Zero steps handled |

