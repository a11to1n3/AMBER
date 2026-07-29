# AAMAS Experimental Improvement Package

Implements the **Experimental improvement plan** from the AAMAS Submission Review
chat (share `t_6a65bb2fcb608191b2c37e98e0254994`).

Evidential chain:

```
declared activation semantics → reference transition → cross-path semantic
validation → runtime-report characterization → validated fast implementation →
fair performance measurement
```

## Claims

| ID | Claim | Primary experiment |
|----|-------|--------------------|
| C1 | Optimized kernels preserve declared reference semantics | `semantic/run_attestation.py` |
| C2 | ContractReport has measurable detection boundary | `monitor/run_coverage.py` |
| C3 | Activation semantics can change ABM conclusions | `benchmarks/run_activation.py` |
| C4 | Speed comparisons fair and reproducible | `benchmarks/run_performance.py` |
| C5 | Performance advantage is explainable | ablations inside performance run |

## Quick start (host_a)

```bash
source <artifact-root>/.venv/bin/activate
export PYTHONPATH=<artifact-root>_aamas_exp/src:<artifact-root>_aamas_exp
cd <artifact-root>_aamas_exp
python experiments/run_all.py --out experiments/raw --tag host_a
```

## Layout

See plan §14. All paper tables/figures for this campaign should be regenerated
from `raw/**/*.json` only.
