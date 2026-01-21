# Comprehensive Correctness Benchmark Report

**Generated**: 2026-01-21 01:20:30

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
| AMBER | 8 | 10 | 80% |
| AgentPy | 8 | 10 | 80% |
| Mesa | 7 | 10 | 70% |

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

### Statistical

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | boltzmann_ks_statistic | ❌ | KS=0.5050, p=0.0000 |
| AMBER | random_walk | diffusion_coefficient_error | ❌ | Relative error: 82965.41% |
| AgentPy | wealth_transfer | boltzmann_ks_statistic | ❌ | KS=0.5350, p=0.0000 |
| AgentPy | random_walk | diffusion_coefficient_error | ❌ | Relative error: 83886.79% |
| Mesa | wealth_transfer | boltzmann_ks_statistic | ❌ | KS=0.4500, p=0.0000 |
| Mesa | random_walk | diffusion_coefficient_error | ❌ | Relative error: 87815.01% |

### Reproducibility

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| AgentPy | wealth_transfer | seed_determinism | ✅ | Same seed → same result |
| Mesa | wealth_transfer | seed_determinism | ❌ | Non-deterministic! |

### Emergent

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | gini_increase | ✅ | Gini change: +0.0581 |
| AMBER | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| AgentPy | wealth_transfer | gini_increase | ✅ | Gini change: +0.0563 |
| AgentPy | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |
| Mesa | wealth_transfer | gini_increase | ✅ | Gini change: +0.0567 |
| Mesa | sir_epidemic | recovery_monotonic | ✅ | R monotonically increasing |

### Precision

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | error_accumulation | ✅ | Error at 100 steps: 0.00e+00, at 1000: 0.00e+00 |
| AgentPy | wealth_transfer | error_accumulation | ✅ | Error at 100 steps: 0.00e+00, at 1000: 0.00e+00 |
| Mesa | wealth_transfer | error_accumulation | ✅ | Error at 100 steps: 0.00e+00, at 1000: 0.00e+00 |

### Edge Case

| Framework | Model | Metric | Status | Details |
|-----------|-------|--------|--------|--------|
| AMBER | wealth_transfer | single_agent | ✅ | Single agent handled |
| AMBER | wealth_transfer | zero_steps | ✅ | Zero steps handled |
| AgentPy | wealth_transfer | single_agent | ✅ | Single agent handled |
| AgentPy | wealth_transfer | zero_steps | ✅ | Zero steps handled |
| Mesa | wealth_transfer | single_agent | ✅ | Single agent handled |
| Mesa | wealth_transfer | zero_steps | ✅ | Zero steps handled |

