# AAMAS paper island (isolated from the AMBER library)

Everything for the AAMAS manuscript and computational artifact lives **here**.
The library package (`src/ambr`, `tests/`, `docs/`, main `benchmarks/`) stays
independent; paper work does not belong on the library install path.

## Layout

```
paper-fix-work/
  README.md                 ← you are here
  artifact/                 ← ANONYMOUS upload package (v6)
    AMBER_AAMAS2027_artifact_v6/
    AMBER_AAMAS2027_artifact_v6.zip
    ARTIFACT_V6_REPAIR.md
  experiments/              ← experiment runners + host_a raw evidence
  tools/host_b_rerun/       ← Host-B campaign orchestration
  manuscript/               ← TeX, bib, figure/repro notes (local drafts OK)
  scripts/                  ← manuscript figure renderers (local)
  evidence/                 ← pointers / optional local campaign pulls
  campaign_results/         ← gitignored: full Host-B pulls (local only)
  figs/                     ← gitignored: local figure drafts
```

## What to upload for AAMAS (anonymous)

**Only** the zip:

```
paper-fix-work/artifact/AMBER_AAMAS2027_artifact_v6.zip
```

Reviewer check (clean machine):

```bash
unzip AMBER_AAMAS2027_artifact_v6.zip
cd AMBER_AAMAS2027_artifact_v6
pip install matplotlib numpy   # or: pip install -r requirements-lock.txt
./reproduce.sh figures
```

## Relationship to the library

| Library tree | Paper island |
|--------------|--------------|
| `src/ambr/` | Artifact embeds a snapshot of `src/` for self-containment |
| `benchmarks/` (library) | Campaign models also copied into artifact; Host-B tooling under `tools/` |
| `tests/` | Not required for figure regen; optional on GPU hosts |

Library-side code fixes needed for experiments (e.g. SIR counter-tape in
`benchmarks/models/`) ship as normal library commits. Paper-only material
stays under `paper-fix-work/`.

## Local-only (gitignored)

- `campaign_results/` — full Host-B campaign trees and logs  
- Overleaf / submission-ready zip trees  
- LaTeX build products (`*.pdf`, `*.aux`, …)  
- `tmp/`, `output/`  

Do **not** put personal hostnames in tracked files; use `host_a` / `host_b`.
