# Comprehensive Correctness Benchmark Report

**Generated**: 2026-01-21 07:19:55

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
| AMBER | 11 | 11 | 100% |
| AgentPy | 11 | 11 | 100% |
| Mesa | 10 | 11 | 91% |

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
| AMBER | wealth_transfer | variance_growth | ✅ | Final variance: 33.12 (expected >0) |
| AMBER | random_walk | mean_squared_displacement | ✅ | MSD: 1338.91 |
| AMBER | sir_epidemic | attack_rate | ✅ | Attack rate: 16.0% |
| AgentPy | wealth_transfer | variance_growth | ✅ | Final variance: 36.56 (expected >0) |
| AgentPy | random_walk | mean_squared_displacement | ✅ | MSD: 1623.87 |
| AgentPy | sir_epidemic | attack_rate | ✅ | Attack rate: 22.0% |
| Mesa | wealth_transfer | variance_growth | ✅ | Final variance: 28.84 (expected >0) |
| Mesa | random_walk | mean_squared_displacement | ✅ | MSD: 1449.48 |
| Mesa | sir_epidemic | attack_rate | ✅ | Attack rate: 18.0% |
| Melodie | wealth_transfer | variance_growth | ✅ | Final variance: 27.16 (expected >0) |
| Melodie | random_walk | mean_squared_displacement | ✅ | MSD: 100.00 |
| Melodie | sir_epidemic | attack_rate | ✅ | Attack rate: 0.0% |
| SimPy | wealth_transfer | variance_growth | ✅ | Final variance: 34.36 (expected >0) |
| SimPy | random_walk | mean_squared_displacement | ✅ | MSD: 100.00 |
| SimPy | sir_epidemic | attack_rate | ✅ | Attack rate: 0.0% |

### Reproducibility

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| AgentPy | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| Mesa | wealth_transfer | seed_determinism | ❌ | Non-deterministic! |
| Melodie | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| SimPy | wealth_transfer | seed_determinism | ✅ | Same seed → same result |

### Emergent

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | gini_increase | ✅ | Gini change: +0.0297 |
| AMBER | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| AgentPy | wealth_transfer | gini_increase | ✅ | Gini change: +0.0315 |
| AgentPy | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| Mesa | wealth_transfer | gini_increase | ✅ | Gini change: +0.0224 |
| Mesa | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| Melodie | wealth_transfer | gini_increase | ✅ | Gini change: +0.0331 |
| Melodie | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| SimPy | wealth_transfer | gini_increase | ✅ | Gini change: +0.0324 |
| SimPy | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |

### Precision

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | error_accumulation | ✅ | Error at 30 steps: 0.00e+00, at 300: 0.00e+00 |
| AgentPy | wealth_transfer | error_accumulation | ✅ | Error at 30 steps: 0.00e+00, at 300: 0.00e+00 |
| Mesa | wealth_transfer | error_accumulation | ✅ | Error at 30 steps: 0.00e+00, at 300: 0.00e+00 |
| Melodie | wealth_transfer | error_accumulation | ✅ | Error at 30 steps: 0.00e+00, at 300: 0.00e+00 |
| SimPy | wealth_transfer | error_accumulation | ✅ | Error at 30 steps: 0.00e+00, at 300: 0.00e+00 |

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

