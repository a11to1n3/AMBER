# Local evidence (not the anonymous artifact)

Full Host-B campaign pulls live in (gitignored):

```
../campaign_results/host_b_rtx5090_20260727T071054Z/
```

Including:

- `HOST_B_DETAILED_RERUN_RESULTS.md`
- `SCOPED_RERUNS.md`
- `benchmark_results_pull/*`
- phase JSON under `02_rng` … `07_performance`

The **submission artifact** already packages the curated subset under
`../artifact/AMBER_AAMAS2027_artifact_v6/`. Re-sync evidence into the artifact
only via an intentional rebuild; do not commit raw campaign logs.
