# Integrating the experimental improvement campaign into the AAMAS manuscript

Source: ChatGPT share *AAMAS Submission Review* (experimental improvement plan).  
Host: `duypham-Z590` (RTX 3090).  
Artifacts: `experiments/raw/REPORT_duypham_z590.*` and JSON under `experiments/raw/`.

## What the campaign closed

The plan’s evidential chain is no longer only rhetorical:

1. **Frozen specs + counter RNG** (`experiments/specs/`, `experiments/rng/`)
2. **Reference transitions + cross-path attestation** (C1: zero mismatches; 7/7 negatives)
3. **ContractReport boundary** (C2: 4/4 public-seam hazards; N,q,c overhead surface)
4. **Activation effects** (C3: SIR sequential vs snapshot; Schelling sequential vs three-stage)
5. **Native performance** (C4: AMBER CPU/GPU + FLAME GPU 2; cold/warm/steady; no trim)

## Manuscript mapping (minimum edits)

| Paper location | What to insert / change |
|---|---|
| §5.1 activation / threshold | Cite **paired counter-RNG SIR** Δ final I at τ≈0.158 with bootstrap CI excluding zero; keep existing large-N controlled experiment as scale extension. |
| §5.3 / monitor cost | Replace “q=0 only” language with **overhead surface** excerpt (at least N∈{1e5,1e6}, c,q variation). State private GPU path is **not** under monitor. |
| Fast-path / private GPU | Add **Table: semantic attestation** — workloads × backends × cases × mismatches × negatives detected. Bind narrative to attestation artifact hash (JSON). |
| §5.4 scaling | Add **AMBER GPU vs FLAME GPU 2** warm medians on wealth (and random walk if measured), with explicit **native-idiom** caveat (not counter-RNG matched). |
| Supplement | Full τ sweep, Schelling disagreement table, full overhead matrix, raw sample schema. |

## Claim language that is now supported

- “All attested AMBER-style backends implementing the frozen snapshot rule matched the reference on exhaustive and random cases; seven intentional corruptions were detected.”
- “ContractReport detected all four injected public-seam hazards in the coverage suite; monitor cost grows with N and is material at 1e6 agents.”
- “Under shared random tape, sequential vs snapshot SIR changes final infected fraction with bootstrap CI excluding zero near the intermediate transmissibility regime.”
- “Three-stage Schelling vs sequential activation disagrees on a positive fraction of cells (primary outcome); happiness-only metrics can be insensitive.”

## Claim language that is still **not** supported

- Byte-identical FLAME ↔ AMBER trajectories (FLAME uses native RNG and different messaging idioms).
- Monitor completeness beyond instrumented seams.
- Universal GPU dominance across all workloads/hardware (single RTX 3090 host).

## Suggested one-paragraph Results insert (draft)

> We close the development-to-deployment chain with a machine-readable transition
> specification and a counter-based random tape shared across reference, vectorized,
> and GPU-style backends. On exhaustive and property-based suites for wealth, random
> walk, SIR, and Schelling, attested backends produced zero discrete state mismatches,
> while seven intentional semantic corruptions diverged. ContractReport recovered all
> four public-seam hazards in a coverage suite; its overhead surface over population
> size, schema width, and write intensity shows multi-fold cost at one million agents,
> so private optimized loops remain outside the monitor. Under the shared tape,
> sequential versus snapshot SIR activation shifts final infected fraction with a
> bootstrap interval excluding zero at intermediate transmissibility, and sequential
> versus three-stage Schelling disagrees on a positive fraction of cells. Native-idiom
> wall-clock comparisons on an RTX 3090 place AMBER’s GPU wealth path against FLAME
> GPU 2 under a single cold/warm protocol without sample trimming.

## Files to cite in reproducibility map

| Claim | Artifact |
|---|---|
| Attestation | `experiments/raw/semantic/attestation_duypham_z590.json` |
| Monitor coverage | `experiments/raw/monitor/coverage_duypham_z590.json` |
| Monitor overhead | `experiments/raw/monitor/overhead_duypham_z590.json` |
| Activation | `experiments/raw/semantic/activation_duypham_z590.json` |
| Performance | `experiments/raw/performance/performance_duypham_z590.json` |
| Summary | `experiments/raw/REPORT_duypham_z590.md` |

## Next manuscript engineering (not done in this campaign)

1. Re-render any table/figure from the new JSON only (no hand-copied numbers).
2. Add attestation SHA256 into `approve_fast_path` evidence string in code listings.
3. Optionally re-run headline 10M AMBER/FLAME campaign with the same cold/warm schema for consistency with C4 (separate from this package’s N≤1e6 native track).
