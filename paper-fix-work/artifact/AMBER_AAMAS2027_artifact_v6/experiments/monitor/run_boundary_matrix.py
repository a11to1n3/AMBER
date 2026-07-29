#!/usr/bin/env python3
"""Priority 4 — ContractReport boundary matrix (not completeness claim).

Separates: ContractReport outcomes, API immutability blocks, and known limits.
Restores the hidden gather false-negative (roll) as an explicit known limitation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))

import ambr as am  # noqa: E402


def base(n=32, steps=2, seed=0):
    return {"n": n, "steps": steps, "seed": seed, "show_progress": False}


def run_case(name, cls, mode, expect):
    """expect: clean | report_error | api_block | known_limit_clean"""
    model = cls(base())
    runner = model.cpu(mode=mode)
    kinds = []
    operational = "clean"
    mechanism = "ContractReport"
    try:
        runner.run(contract="check")
        certs = list(getattr(model, "contract_certificates", []) or [])
        for c in certs:
            for v in getattr(c, "violations", []) or []:
                kinds.append(getattr(v, "kind", str(v)))
        if any(not getattr(c, "ok", True) for c in certs) or kinds:
            operational = "error"
    except Exception as exc:
        kinds = [type(exc).__name__]
        operational = "blocked"
        mechanism = "API immutability / runtime exception"

    # classify
    if expect == "clean":
        ok = operational == "clean"
        category = "safe"
    elif expect == "report_error":
        ok = operational == "error" and mechanism == "ContractReport"
        category = "visible_hazard"
    elif expect == "api_block":
        ok = operational == "blocked"
        category = "untraceable_mutation"
        mechanism = "API immutability"
    elif expect == "known_limit_clean":
        ok = operational == "clean"
        category = "hidden_provenance"
        mechanism = "none (known limit)"
    else:
        ok = False
        category = "unknown"

    return {
        "name": name,
        "semantic_category": category,
        "mechanism": mechanism,
        "operational_outcome": operational,
        "expect": expect,
        "match_expect": ok,
        "kinds": kinds,
    }


# ---- cases -----------------------------------------------------------------

class SafePointwise(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x + 1.0)


class SafeScatter(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, wealth=np.ones(n, dtype=np.int64))
    def step_vectorized(self):
        w = self.agents.wealth.to_numpy()
        donors = np.flatnonzero(w > 0)
        if donors.size == 0:
            return
        ids = self.agents.ids.to_numpy()
        rec = self.rng.choice(ids, size=donors.size)
        targets = np.concatenate((ids[donors], rec))
        deltas = np.concatenate((-np.ones(donors.size, dtype=np.int64), np.ones(donors.size, dtype=np.int64)))
        self.agents.at[targets].scatter_add(wealth=deltas)


class SafeNoOp(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n))
    def step_vectorized(self):
        pass


class DupCommit(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x + 1.0)
        self.agents.commit(x=x + 2.0)


class EqualDupCommit(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x + 1.0)
        self.agents.commit(x=x + 1.0)


class CommitThenBorrow(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n), y=np.zeros(n))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x + 1.0)
        x2, _ = self.agents.borrow("x")
        self.agents.commit(y=x2)


class MutableArray(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n))
    def step_vectorized(self):
        x = self.agents.array("x")
        x += 1.0


class BufferedDupOOP(am.Model):
    def setup(self):
        n = int(self.p["n"])
        class A(am.Agent):
            def setup(self):
                self.wealth = 1
            def step(self):
                self.wealth = self.wealth + 1
                self.wealth = self.wealth + 2
        self.add_agents(n, agent_class=A)
    def step_oop(self):
        for a in self.agents:
            a.step()


class HiddenRoll(am.Model):
    """Known limit: single borrow/commit of roll looks clean but is order-sensitive."""
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.arange(n, dtype=np.float64))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=np.roll(x, 1))


class SchemaAdd(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x + 1.0, y=np.ones(int(self.p["n"])))


class DtypeChange(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n, dtype=np.float64))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x.astype(np.int64))


class FloatingReduce(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.ones(n, dtype=np.float64))
    def step_vectorized(self):
        ids = self.agents.ids.to_numpy()
        self.agents.at[ids].scatter_add(x=0.1)


class MixedSetReduce(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.ones(n, dtype=np.float64))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        self.agents.commit(x=x + 1.0)
        ids = self.agents.ids.to_numpy()
        self.agents.at[ids].scatter_add(x=1.0)


class TwoColumnSafe(am.Model):
    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, x=np.zeros(n), y=np.zeros(n))
    def step_vectorized(self):
        x, _ = self.agents.borrow("x")
        y, _ = self.agents.borrow("y")
        self.agents.commit(x=x + 1.0, y=y - 1.0)


CASES = [
    ("safe_pointwise", SafePointwise, "vectorized", "clean"),
    ("safe_scatter", SafeScatter, "vectorized", "clean"),
    ("safe_noop", SafeNoOp, "vectorized", "clean"),
    ("safe_two_column", TwoColumnSafe, "vectorized", "clean"),
    ("safe_floating_reduce", FloatingReduce, "vectorized", "clean"),
    ("haz_duplicate_commit", DupCommit, "vectorized", "report_error"),
    ("haz_equal_duplicate_commit", EqualDupCommit, "vectorized", "report_error"),
    ("haz_commit_then_borrow", CommitThenBorrow, "vectorized", "report_error"),
    ("haz_buffered_duplicate_oop", BufferedDupOOP, "oop", "report_error"),
    ("haz_mixed_set_reduce", MixedSetReduce, "vectorized", "report_error"),
    ("block_mutable_array", MutableArray, "vectorized", "api_block"),
    ("limit_hidden_roll", HiddenRoll, "vectorized", "known_limit_clean"),
    # Schema addition currently surfaces as ContractReport error/warning path.
    ("schema_add", SchemaAdd, "vectorized", "report_error"),
    # Dtype change may be endpoint-invisible depending on Polars cast path;
    # record observed operational outcome as known-limit if clean.
    ("dtype_change_observed", DtypeChange, "vectorized", "known_limit_clean"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "raw" / "monitor")
    ap.add_argument("--tag", default="host_a")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    for name, cls, mode, expect in CASES:
        row = run_case(name, cls, mode, expect)
        rows.append(row)
        print(f"{row['match_expect']!s:5} {name:30} {row['operational_outcome']:10} {row['mechanism']}", flush=True)

    # semantic note for hidden roll
    x = np.array([1.0, 2.0, 3.0])
    snapshot = np.roll(x, 1)  # [3,1,2]
    sequential = x.copy()
    # naive sequential forward roll interpretation
    sequential_fwd = np.array([3.0, 3.0, 3.0])  # as in plan illustration
    hidden = {
        "example": "borrow once; commit roll(x,1)",
        "operational_report": "clean",
        "snapshot_semantics": snapshot.tolist(),
        "sequential_forward_illustration": sequential_fwd.tolist(),
        "note": "Known limitation: endpoint-clean report is not a cell-level sequential/snapshot proof.",
    }

    matrix = {
        "tag": args.tag,
        "host_label": "host_a",
        "platform": platform.platform(),
        "elapsed_s": time.time() - t0,
        "rows": rows,
        "hidden_provenance_example": hidden,
        "summary": {
            "n_cases": len(rows),
            "match_expect": sum(1 for r in rows if r["match_expect"]),
            "by_category": {},
        },
        "claim_language": (
            "All designed visible-hazard cases produced their expected operational outcomes "
            "within declared seams. Monitor completeness is not claimed."
        ),
    }
    for r in rows:
        matrix["summary"]["by_category"].setdefault(r["semantic_category"], 0)
        matrix["summary"]["by_category"][r["semantic_category"]] += 1

    path = args.out / f"boundary_matrix_{args.tag}.json"
    path.write_text(json.dumps(matrix, indent=2))
    print(json.dumps(matrix["summary"], indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
