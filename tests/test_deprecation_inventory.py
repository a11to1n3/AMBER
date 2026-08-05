"""Ensure the 1.0 deprecation inventory matches warn_deprecated call sites."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from ambr.deprecation_inventory import DEPRECATIONS_TO_REMOVE_IN_1_0, inventory_whats

SRC = Path(__file__).resolve().parents[1] / "src" / "ambr"


def _extract_warn_deprecated_whats() -> set[str]:
    """Parse src/ambr for warn_deprecated( first positional string arg."""
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path.name.startswith("_") and path.name not in (
            "_deprecation.py",
        ):
            # still scan private modules that emit warnings
            pass
        if path.name == "deprecation_inventory.py":
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name) and func.id == "warn_deprecated":
                name = "warn_deprecated"
            elif isinstance(func, ast.Attribute) and func.attr == "warn_deprecated":
                name = "warn_deprecated"
            if name != "warn_deprecated":
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                found.add(arg0.value)
    return found


@pytest.mark.unit
def test_inventory_covers_all_warn_deprecated_sites():
    sites = _extract_warn_deprecated_whats()
    # decorator builds dynamic names — ignore empty
    inventory = set(inventory_whats())
    missing = sites - inventory
    extra = inventory - sites
    assert not missing, (
        f"warn_deprecated sites not in deprecation_inventory: {sorted(missing)}\n"
        "Add them to DEPRECATIONS_TO_REMOVE_IN_1_0."
    )
    assert not extra, (
        f"inventory entries with no warn_deprecated site: {sorted(extra)}\n"
        "Remove stale rows or fix the call-site string."
    )


@pytest.mark.unit
def test_inventory_non_empty_and_stable_count():
    assert len(DEPRECATIONS_TO_REMOVE_IN_1_0) >= 15
    # create_batch_context uses warnings.warn directly — still documented in docs
    assert any("Experiment" in w for w in inventory_whats())
