#!/usr/bin/env python3
"""Reconcile the all-framework sweep with the final corrected 10M campaign.

The historical sweep supplies coverage across ten timed implementations and
five population sizes.  The final campaign is authoritative for the eight
AMBER-GPU/FLAME-GPU-2 rows at 10 million agents.  This script replaces exactly
those endpoints, preserves historical row order, and records every old/new
value in the output artifact.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
HISTORICAL = ARTIFACTS / "benchmark_results_all5090.json"
FINAL = ARTIFACTS / "benchmark_results_snapshot_correct_10run_10m.json"
OUTPUT = ARTIFACTS / "benchmark_results_all5090_reconciled.json"

MODELS = {"wealth_transfer", "random_walk", "sir_epidemic", "schelling"}
FINAL_FRAMEWORKS = {"AMBER (GPU)", "FLAME GPU 2"}


def key(row: dict) -> tuple[str, str, int]:
    return row["framework"], row["model"], int(row["n_agents"])


def main() -> None:
    historical = json.loads(HISTORICAL.read_text())
    final = json.loads(FINAL.read_text())

    historical_rows = historical["results"]
    final_rows = final["results"]

    historical_keys = [key(row) for row in historical_rows]
    assert len(historical_rows) == 142
    assert len(historical_keys) == len(set(historical_keys))
    assert all(row["execution_time"] > 0 for row in historical_rows)
    assert set(historical["agent_counts"]) == {1_000, 10_000, 100_000, 1_000_000, 10_000_000}

    final_by_key = {key(row): row for row in final_rows}
    expected_final = {
        (framework, model, 10_000_000)
        for framework in FINAL_FRAMEWORKS
        for model in MODELS
    }
    assert set(final_by_key) == expected_final
    assert expected_final.issubset(set(historical_keys))
    assert all(row["raw_sample_count"] == 10 for row in final_rows)
    assert all(row["trimmed"] == 0 for row in final_rows)

    reconciled_rows: list[dict] = []
    replacements: list[dict] = []
    for old_row in historical_rows:
        row_key = key(old_row)
        if row_key in final_by_key:
            new_row = dict(final_by_key[row_key])
            new_row["source_campaign"] = "final corrected ten-run 10M campaign"
            replacements.append(
                {
                    "framework": row_key[0],
                    "model": row_key[1],
                    "n_agents": row_key[2],
                    "historical_execution_time": old_row["execution_time"],
                    "final_execution_time": new_row["execution_time"],
                }
            )
        else:
            new_row = dict(old_row)
            new_row["source_campaign"] = "historical all-framework scaling sweep"
        reconciled_rows.append(new_row)

    assert len(reconciled_rows) == len(historical_rows)
    assert len(replacements) == 8
    assert [key(row) for row in reconciled_rows] == historical_keys

    output = dict(historical)
    output["results"] = reconciled_rows
    output["reconciliation"] = {
        "historical_source": HISTORICAL.name,
        "authoritative_endpoint_source": FINAL.name,
        "rule": (
            "Preserve every historical row except AMBER (GPU) and FLAME GPU 2 "
            "at N=10,000,000; replace those eight rows with the final corrected "
            "ten-run arithmetic-mean records."
        ),
        "historical_timing_scope": historical.get("timing"),
        "final_endpoint_timing": final.get("timing"),
        "replacements": replacements,
    }
    output["note"] = (
        f"{historical.get('note', '')} | Figure-5 reconciliation: the eight "
        "AMBER-GPU/FLAME-GPU-2 endpoints at 10M are superseded by the final "
        "corrected ten-run campaign; all other rows are unchanged."
    )

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {OUTPUT.name}: {len(reconciled_rows)} rows, {len(replacements)} replacements")


if __name__ == "__main__":
    main()
