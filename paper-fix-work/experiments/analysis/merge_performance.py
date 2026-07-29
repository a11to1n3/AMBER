#!/usr/bin/env python3
"""Merge performance JSON artifacts, preferring newer native rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def key(r):
    return (r.get("track"), r.get("workload"), r.get("population"), r.get("framework"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = json.loads(args.base.read_text()) if args.base.exists() else {"rows": []}
    extra = json.loads(args.extra.read_text()) if args.extra.exists() else {"rows": []}
    by = {}
    for r in base.get("rows") or []:
        by[key(r)] = r
    for r in extra.get("rows") or []:
        by[key(r)] = r  # overwrite/add
    rows = list(by.values())
    # recompute amber_vs_flame
    med = {}
    for r in rows:
        if r.get("status") != "success" or r.get("track") != "native_idiom":
            continue
        med.setdefault((r.get("workload"), r.get("population")), {})[r["framework"]] = r.get("warm_median_s")
    avf = []
    for (wl, n), d in sorted(med.items()):
        if d.get("AMBER_gpu") and d.get("FLAME_GPU_2"):
            avf.append({
                "workload": wl,
                "population": n,
                "AMBER_gpu_s": d["AMBER_gpu"],
                "FLAME_GPU_2_s": d["FLAME_GPU_2"],
                "amber_over_flame": d["AMBER_gpu"] / d["FLAME_GPU_2"] if d["FLAME_GPU_2"] else None,
                "flame_over_amber": d["FLAME_GPU_2"] / d["AMBER_gpu"] if d["AMBER_gpu"] else None,
            })
    out = {
        "tag": extra.get("tag") or base.get("tag"),
        "host": extra.get("host") or base.get("host"),
        "platform": extra.get("platform") or base.get("platform"),
        "merged_from": [str(args.base), str(args.extra)],
        "cupy": extra.get("cupy", base.get("cupy")),
        "ambr": extra.get("ambr", base.get("ambr")),
        "flame": extra.get("flame", base.get("flame")),
        "rows": rows,
        "speedups_gpu_vs_reference": base.get("speedups_gpu_vs_reference") or extra.get("speedups_gpu_vs_reference"),
        "amber_vs_flame": avf,
        "notes": "Merged base (matched) + extra (native/flame). No samples trimmed.",
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"merged {len(rows)} rows, amber_vs_flame={len(avf)} -> {args.out}")

if __name__ == "__main__":
    main()
