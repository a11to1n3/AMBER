# Claim dashboard (honest status)

Host labels: `host_a` (RTX 3090 semantic/attestation package) and `host_b`
(RTX 5090 multi-framework package). Paths use `<artifact-root>/`.

| Claim | Status |
|---|---|
| Reference / vectorized NumPy / GPU-style parity | **passed** |
| Production native-kernel parity — wealth | **passed** (0 mismatches, 310 cases) |
| Production native-kernel parity — random walk | **passed** (0 mismatches, 310 cases; float32 tol 1e−4) |
| Production native-kernel parity — Schelling | **passed** (0 mismatches, 58 cases) |
| Production native-kernel parity — SIR | **passed** (0 mismatches, 289 cases; pair-keyed SplitMix64 in CUDA join) |
| Negative-control sensitivity | **passed** within designed faults |
| Report boundary scenarios | **passed within declared seams** |
| Monitor completeness | **not claimed** |
| SIR cumulative attack-rate crossing under shared RNG | **passed** (τ_c@0.5 shift +0.034 [0.024, 0.043]; ~14.2% relative) |
| Multi-framework performance | **measured under two documented campaigns** |

## Submission gate

- [x] Exact timed private kernels: **zero state mismatches** for wealth, random walk, Schelling, **and SIR**
- [x] Shared-RNG cumulative-attack SIR crossing with paired interval excluding zero (primary A=0.5)
- [x] Historical scaling and final endpoint campaigns are **not** one statistical trajectory (Panel A / Panel B)

## Terminology (manuscript v4)

| Avoid | Prefer |
|---|---|
| approval-gated | **caller-attested** |
| validated fast implementation | **production-kernel attested** (per workload) |
| all hazards detected | designed visible-hazard cases → expected operational outcomes |
| cold | first invocation in campaign process |
| experimental \(q\) for column commits | \(k\) = observed column commits per step |
