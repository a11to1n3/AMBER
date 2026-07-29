# AMBER AAMAS manuscript review report

## Outcome

The paper has been reconciled against the current AMBER implementation and the
released raw artifacts. The revised framing now distinguishes four things that
must not be conflated: the sufficient event-level theorem, the bounded runtime
diagnostic, the standalone semantic experiments, and the private optimized GPU
benchmark path.

## Material corrections made

- Rewrote the title, abstract, introduction, contribution list, related work,
  discussion, and conclusion around one four-layer argument: formal scope,
  observable runtime evidence, controlled semantic effects, and an
  approval-gated throughput path whose evidence is stated separately.
- Made the theorem's quantifiers explicit. Events, targets, read sets, and
  order-independent random draws are fixed at step entry; dynamic same-step
  event generation is outside the result unless frozen or staged first.
- Replaced the ambiguous snapshot reduction case with a total cell-level case
  split: zero events, one set, one reduction, repeated reductions through one AC
  operator, or conflict. The non-interference definition is now explicitly
  symmetric, and the proof accounts for an event reading its own target.
- Clarified that the controlled in-place SIR program changes same-step event
  opportunities and is therefore a model-level activation comparison, not a
  direct theorem instance. Overlapping snapshot intervals are described as a
  consistency check rather than an equivalence test. The headline schedule now
  reshuffles the coupled activation order at every step and reports a 6.9%
  shift; the previous 12.1% fixed-order result is retained only as robustness.
- Defined temporary double-buffer storage modulo projection onto the observable
  model state, coupled trajectory randomness by stable event/agent identity,
  and stated expected monitor time and space as `O(N + q + c)` with `c` equal
  to schema columns checked at the step boundary.
- Aligned the implementation narrative with the code: cross-seam writes,
  endpoint-only structural checks, mutate-and-revert limitations, and the exact
  behavior of `check`, `warn`, and `raise` are now stated.
- Expanded Section 4 into an explicit development/deployment methodology. New
  Algorithm 1 formalizes mode resolution, fast-path eligibility, explicit
  caller approval, per-step monitor setup/finalization, contract policy
  application, and a shared synchronized teardown. The implementation now
  exposes `approve_fast_path(evidence)` and ignores private hooks without a
  non-empty per-instance label; AMBER does not claim to validate that label.
- Replaced the invalid theorem converse with a sufficient-only statement and
  retained explicit counterexamples to necessity.
- Limited monitor claims to operations visible at instrumented API seams; a
  clean trace is no longer described as a semantic proof.
- Corrected the mesa-frames study: all variants make one final framework `set`;
  only local NumPy update-block granularity changes.
- Replaced the old wealth GPU kernel with frozen donor eligibility and aligned
  OOP, vectorized CPU, and GPU descriptions with the actual native paths.
- Replaced the failed FLAME SIR row with the successful RTC execution and
  updated all four 10M rows from the final ten-run artifact.
- Added raw-sample retention, provenance, uncertainty summaries, and explicit
  implementation-comparison caveats.
- Added the exact SIR fast-path complexity caveat: `O(N + C + P)` with a
  quadratic worst case under growing fixed-domain density, not guaranteed
  `O(N)`.
- Corrected stale supplementary scripts that still claimed theorem necessity
  and a general barrier lower bound.
- Rebuilt the topological-staging figure so its title, axes, legend, and
  caption all describe the bounded sequential-reference control rather than a
  lower-bound theorem.
- Rebuilt every included figure under one print-oriented editorial system with
  Avenir/DejaVu typography, Atlantic-blue and copper semantic roots, warmer
  neutrals, restrained grids, and vector PDF output. Removed legend/title
  collisions, crowded panel labels, and column-width three-panel layouts.
- Rebuilt Figure 1 as a restrained vector schematic rather than a generated
  illustration. AMBER's three exact API modes occupy the centre between quiet
  agent-centric and array/accelerator example groups; monitor/reference scope
  and the non-universality caveat remain explicit, and the private loop is
  marked approval-gated rather than reference-validated.
- Rebuilt Figure 2 as a conventional decision-driven flowchart rather than a
  staged infographic or unordered lane diagram. Standard process, decision,
  and terminator shapes now expose the contract-hazard loop, external evidence,
  caller approval, fast-path eligibility, general/private runtime split, and a
  synchronized common exit. Exact native API labels remain visible, approval
  is not presented as validation, and native `gpu()` is not conflated with the
  private fast path. The conservative P/G/R/U taxonomy remains in Appendix A.
- Rebuilt Figure 5 as a four-workload, log--log scaling landscape containing
  all ten timed implementations. AMBER and FLAME GPU 2 carry the visual focus;
  the remaining frameworks stay identifiable as neutral context, and direct
  labels report each final speedup. Its eight outlined AMBER-GPU/FLAME-GPU-2
  10M endpoints come from the final corrected campaign and connect to the
  historical trajectories; outlined markers disclose the aggregation change.
  Schelling is marked setup-inclusive; AgentTorch is
  capability-only because no matched timing row was recorded.
- Expanded the cited bibliography from 21 to 35 works without padding. The
  revision now sources every timed framework, adds foundational activation-
  regime and GPU-ABM work, and strengthens the dependence-detection,
  distributed-simulation, runtime-verification, and benchmarking context. All
  35 bibliography entries are cited in the manuscript.
- Added an executable reconciliation audit that validates 142 historical rows,
  replaces exactly the eight authoritative endpoint keys, checks all ten final
  samples are retained, and records every old/new timing value.
- Split the SLOC and calibration artifacts into separate full-width
  supplementary figures.
- Added `FIGURE_MANIFEST.md` and one deterministic renderer for all included
  figures, with an explicit source artifact and chart contract for each plot.
- Removed a forced appendix page break that left most of one page blank.
- Polished the supplement as a continuous publication artifact: added a formal
  roadmap, pinned the rule-taxonomy table before the commuting-reducer result,
  repaired display punctuation, removed full-page vertical stretching around
  evidence figures, clarified how the supporting diagnostics should be read,
  aligned the three native AMBER API paths in the execution table, and kept
  every framework listing intact across page boundaries.
- Corrected bibliography metadata, figure descriptions, code-listing drift,
  and figure labels that could imply framework-level block commits.

## Evidence checks

- Final 10M means reproduce from all ten retained samples: AMBER/FLAME speedups
  are 2.05x (wealth), 2.00x (random walk), 1.77x (SIR), and 63.4x (Schelling).
  The headline range is 1.77x--2.05x; Schelling is exploratory because timed
  FLAME setup includes a Python per-agent initialization loop.
- With per-step reshuffling, the paired SIR shift is 0.0167 [0.0108, 0.0236],
  or 6.9% of the row-wise finite-horizon crossing estimate. The fixed-order
  robustness condition is 0.0295 [0.0195, 0.0390], or 12.1%.
- Row-wise and GPU snapshot intervals overlap; the 100k GPU row is presented
  only as a scale check.
- The theorem referee reports 0 violations among 1,327 generated
  non-interfering event sets and 34 interfering sets without a finite-grid
  divergence witness; the latter are controls against a converse, not cases to
  discard.
- The current monitor microbenchmark is explicitly restricted to `q=0`
  whole-column commits and is not compared with the private GPU loop.
- A fresh bounded theorem run reproduced 1,327 non-interfering cases with zero
  violations, 6,000 premise-satisfying trajectories with zero mismatches, all
  720 orders of the six-cell double-buffer control, and the commuting
  subtraction control.

## Software verification

- Repository suite: 419 passed and 15 skipped in the sandbox; the only two
  failures required process semaphores denied there. Both multiprocessing tests
  passed with normal host permissions, so all 421 collected non-skipped tests
  passed.
- Targeted monitor/execution/TensorLane/GPU suite: 57 passed and 10
  GPU-dependent tests skipped.
- `git diff --check`, Python byte-compilation, source-snapshot equality for the
  changed implementation/tests/runner, and every released JSON parse passed.

## Final build and visual QA

- `tectonic --keep-logs --keep-intermediates amber_aamas.tex` completed
  successfully.
- The final log has no overfull boxes or unresolved references/citations.
- The combined artifact is 21 pages. The review manuscript is nine pages: an
  eight-page body followed by one references page. The 12-page supplement
  starts at Appendix A.
- All 21 pages were rendered at 160 dpi and visually inspected. Result plots,
  the full-width method figure, revised Algorithm 1, DAG control, tables, and
  all framework listings were additionally checked at full rendered resolution.
  Appendix A's proof/table continuity, Appendix C's figure spacing, and
  Appendix D's listing pagination were checked page by page.
- The deterministic 160-file ZIP is 2.8 MB, contains byte-identical copies of
  the three PDFs, excludes unused generated-image files and package caches,
  normalizes member metadata, and passes the private-marker scan.

## Author-only submission actions

1. Replace `\submissionid{TBD}` with the assigned venue paper-tracking number.
   The placeholder is visible in the author position and running header.
2. Review and adapt `AI_ASSISTANCE_DISCLOSURE.md` to the venue's current policy.
3. Upload the nine-page main PDF (eight content pages plus references) and the
   12-page appendix PDF as supplementary material if the venue requires them
   to be separate. The full combined PDF is retained for archival review.
