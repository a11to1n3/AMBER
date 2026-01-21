# Multi-Framework Correctness Benchmark

**Generated**: 2026-01-21 01:09:23

## Summary by Framework

| Framework | Tests Passed | Pass Rate |
|-----------|--------------|-----------|
| ✅ AMBER | 3/3 | 100% |
| ✅ AgentPy | 3/3 | 100% |
| ✅ Mesa | 3/3 | 100% |
| ✅ Melodie | 1/1 | 100% |
| ✅ SimPy | 1/1 | 100% |

## Detailed Results by Test

### Wealth Transfer

| Framework | Test | Status | Time | Details |
|-----------|------|--------|------|--------|
| AMBER | conservation | ✅ | 0.0108s | Expected: 10000, Got: 10000 |
| AgentPy | conservation | ✅ | 0.0110s | Expected: 10000, Got: 10000 |
| Mesa | conservation | ✅ | 0.0221s | Expected: 10000, Got: 10000 |
| Melodie | conservation | ✅ | 0.0051s | Expected: 10000, Got: 10000 |
| SimPy | conservation | ✅ | 0.0051s | Expected: 10000, Got: 10000 |

### Sir Epidemic

| Framework | Test | Status | Time | Details |
|-----------|------|--------|------|--------|
| AMBER | population_conservation | ✅ | 0.0046s | S=84, I=0, R=16, Total=100 |
| AgentPy | population_conservation | ✅ | 0.0062s | S=88, I=0, R=12, Total=100 |
| Mesa | population_conservation | ✅ | 0.0067s | S=82, I=0, R=18, Total=100 |

### Random Walk

| Framework | Test | Status | Time | Details |
|-----------|------|--------|------|--------|
| AMBER | spatial_bounds | ✅ | 0.0023s | Out of bounds: 0 |
| AgentPy | spatial_bounds | ✅ | 0.0036s | Out of bounds: 0 |
| Mesa | spatial_bounds | ✅ | 0.0038s | Out of bounds: 0 |

## Scientific Invariants Tested

| Model | Invariant | Description |
|-------|-----------|-------------|
| Wealth Transfer | Conservation | Total wealth S = constant |
| SIR Epidemic | Conservation | S + I + R = N at all times |
| Random Walk | Boundedness | All agents stay within world limits |
