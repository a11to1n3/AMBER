"""RunResults save/load (manifest v1) and Experiment constructor polish."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import ambr as am
import polars as pl
import pytest

from ambr.contract import ContractCertificate, ContractViolation
from ambr.results import RunResults, RunResultsIOError, SCHEMA_VERSION, _sha256_file


class _Tiny(am.Model):
    model_reporters = {"total": lambda m: int(m.agents.wealth.sum())}

    def setup(self):
        self.add_agents(15, wealth=1)

    def step_vectorized(self):
        donors = self.agents.where(self.agents.wealth > 0)
        donors.wealth -= 1
        rec = self.rng.choice(self.agents.ids.to_numpy(), size=len(donors))
        self.agents.at[rec].scatter_add(wealth=1)


@pytest.mark.unit
def test_runresults_save_load(tmp_path):
    r = _Tiny({"steps": 5, "seed": 0, "show_progress": False}).run()
    assert isinstance(r, am.RunResults)
    overview = r.keys_overview()
    assert "model" in overview and "agents" in overview

    dest = tmp_path / "run0"
    r.save(dest)
    assert (dest / "manifest.json").is_file()
    assert (dest / "frames").is_dir()
    assert (dest / "json").is_dir()

    loaded = am.RunResults.load(dest)
    assert loaded.model.height == r.model.height
    assert loaded.agents.height == r.agents.height
    assert loaded.info.get("steps") == r.info.get("steps")


@pytest.mark.unit
def test_manifest_schema_and_opaque_paths(tmp_path):
    r = RunResults(
        {
            "info": {"steps": 1},
            "model": pl.DataFrame({"t": [1], "x": [2.0]}),
            "../escape": pl.DataFrame({"a": [1]}),
            "nested/evil": 42,
        }
    )
    dest = tmp_path / "safe"
    r.save(dest)

    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert set(manifest["entries"]) == {"info", "model", "../escape", "nested/evil"}

    for key, entry in manifest["entries"].items():
        rel = entry["file"]
        # User-controlled keys must never appear as path components.
        assert ".." not in Path(rel).parts
        assert key not in rel
        assert "/" not in key or key not in rel
        assert Path(rel).is_absolute() is False
        full = dest / rel
        assert full.is_file()
        assert full.resolve().is_relative_to(dest.resolve())


@pytest.mark.unit
def test_path_escape_cannot_leave_destination(tmp_path):
    """Even a corrupt manifest with ../ cannot read/write outside the run dir."""
    dest = tmp_path / "run"
    dest.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("top-secret\n")

    # Craft a malicious manifest pointing at ../secret.txt
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "entries": {
            "stolen": {
                "kind": "json",
                "format": "json",
                "file": "../secret.txt",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RunResultsIOError, match="unsafe|escape"):
        RunResults.load(dest)


@pytest.mark.unit
def test_dataframe_then_scalar_same_key_returns_scalar(tmp_path):
    r = RunResults()
    r["metric"] = pl.DataFrame({"v": [1, 2, 3]})
    r["metric"] = 99  # scalar overwrites frame in the mapping
    dest = tmp_path / "scalar_wins"
    r.save(dest)
    loaded = RunResults.load(dest)
    assert loaded["metric"] == 99
    assert not isinstance(loaded["metric"], pl.DataFrame)


@pytest.mark.unit
def test_removed_keys_do_not_reappear(tmp_path):
    dest = tmp_path / "prune"
    r = RunResults(
        {
            "keep": 1,
            "drop_me": pl.DataFrame({"x": [1]}),
            "also_drop": "gone",
        }
    )
    r.save(dest)
    assert set(RunResults.load(dest)) == {"keep", "drop_me", "also_drop"}

    r2 = RunResults({"keep": 2, "new": "fresh"})
    r2.save(dest)  # full rewrite of manifest; orphans may remain on disk
    loaded = RunResults.load(dest)
    assert set(loaded) == {"keep", "new"}
    assert loaded["keep"] == 2
    assert "drop_me" not in loaded
    assert "also_drop" not in loaded


@pytest.mark.unit
def test_interrupted_save_leaves_previous_manifest_readable(tmp_path):
    dest = tmp_path / "atomic"
    first = RunResults({"info": {"v": 1}, "model": pl.DataFrame({"t": [0]})})
    first.save(dest)
    original_manifest = (dest / "manifest.json").read_text()

    # Simulate a crash after writing new payload files but before replace:
    # write a new frame + a .tmp manifest that is NOT committed.
    stem = "deadbeefcafebabe"
    frames = dest / "frames"
    frames.mkdir(exist_ok=True)
    new_frame = frames / f"{stem}.parquet"
    pl.DataFrame({"t": [99]}).write_parquet(new_frame)
    tmp_manifest = {
        "schema_version": SCHEMA_VERSION,
        "entries": {
            "model": {
                "kind": "dataframe",
                "format": "parquet",
                "file": f"frames/{stem}.parquet",
                "sha256": hashlib.sha256(new_frame.read_bytes()).hexdigest(),
            },
            "info": {
                "kind": "json",
                "format": "json",
                "file": "json/should_not_load.json",
                "sha256": "0" * 64,
            },
        },
    }
    (dest / "manifest.json.tmp").write_text(json.dumps(tmp_manifest))
    # Previous committed manifest must still be intact
    assert (dest / "manifest.json").read_text() == original_manifest

    loaded = RunResults.load(dest)
    assert loaded["info"] == {"v": 1}
    assert loaded["model"]["t"].to_list() == [0]


@pytest.mark.unit
def test_corrupt_file_fails_clearly(tmp_path):
    dest = tmp_path / "corrupt"
    r = RunResults({"model": pl.DataFrame({"t": [1, 2]})})
    r.save(dest)

    manifest = json.loads((dest / "manifest.json").read_text())
    rel = manifest["entries"]["model"]["file"]
    target = dest / rel
    # Flip a byte so the checksum no longer matches
    raw = bytearray(target.read_bytes())
    raw[0] = (raw[0] + 1) % 256
    target.write_bytes(bytes(raw))

    with pytest.raises(RunResultsIOError, match="sha256|Corrupt"):
        RunResults.load(dest)


@pytest.mark.unit
def test_missing_file_incomplete_save(tmp_path):
    dest = tmp_path / "incomplete"
    r = RunResults({"model": pl.DataFrame({"t": [1]})})
    r.save(dest)
    manifest = json.loads((dest / "manifest.json").read_text())
    rel = manifest["entries"]["model"]["file"]
    (dest / rel).unlink()
    with pytest.raises(RunResultsIOError, match="Incomplete|missing"):
        RunResults.load(dest)


@pytest.mark.unit
def test_legacy_0_4_x_directory_still_loads(tmp_path):
    """Pre-manifest layout: info.json + model.parquet at root."""
    dest = tmp_path / "legacy"
    dest.mkdir()
    (dest / "info.json").write_text(json.dumps({"steps": 3, "seed": 0}) + "\n")
    pl.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0]}).write_parquet(
        dest / "model.parquet"
    )
    pl.DataFrame({"id": [0, 1], "wealth": [1, 2]}).write_parquet(dest / "agents.parquet")
    (dest / "extra.json").write_text(json.dumps({"note": "legacy"}) + "\n")
    (dest / "contract_summary.json").write_text(
        json.dumps([{"step": 1, "ok": True, "n_violations": 0}]) + "\n"
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded = RunResults.load(dest)
    assert any("legacy" in str(x.message).lower() for x in w)
    assert loaded.info["steps"] == 3
    assert loaded.model.height == 3
    assert loaded.agents.height == 2
    assert loaded["note"] == "legacy"
    assert "contract_summary" in loaded


@pytest.mark.unit
def test_contract_full_violations_persisted(tmp_path):
    cert = ContractCertificate(step=3)
    cert.add(
        ContractViolation(
            kind="duplicate_write",
            detail="cell written twice",
            severity="error",
            columns=["wealth"],
            ids=[0, 1],
        )
    )
    cert.add(
        ContractViolation(
            kind="schema_mutation",
            detail="column removed",
            severity="warning",
            columns=["x"],
        )
    )
    r = RunResults({"contract": [cert], "info": {"steps": 3}})
    dest = tmp_path / "contract_full"
    r.save(dest)
    loaded = RunResults.load(dest)
    assert "contract" in loaded
    payload = loaded["contract"]
    assert isinstance(payload, list) and len(payload) == 1
    c0 = payload[0]
    assert c0["step"] == 3
    assert c0["ok"] is False
    assert c0["clean"] is False
    assert len(c0["violations"]) == 2
    kinds = {v["kind"] for v in c0["violations"]}
    assert kinds == {"duplicate_write", "schema_mutation"}
    dup = next(v for v in c0["violations"] if v["kind"] == "duplicate_write")
    assert dup["columns"] == ["wealth"]
    assert dup["ids"] == [0, 1]
    assert "cell written twice" in dup["detail"]


@pytest.mark.unit
def test_manifest_requires_sha256(tmp_path):
    dest = tmp_path / "nosha"
    r = RunResults({"x": 1})
    r.save(dest)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["entries"].values():
        entry.pop("sha256", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RunResultsIOError, match="sha256"):
        RunResults.load(dest)


@pytest.mark.unit
def test_json_payload_checksum_stable_with_newlines(tmp_path):
    """LF JSON bytes must round-trip checksums on all platforms (incl. Windows)."""
    r = RunResults({"note": "line1\nline2\n", "n": 1})
    dest = tmp_path / "lf"
    r.save(dest)
    # Load re-hashes every payload; mismatch would raise RunResultsIOError.
    loaded = RunResults.load(dest)
    assert loaded["note"] == "line1\nline2\n"
    assert loaded["n"] == 1
    # On-disk bytes must match the hash in the manifest exactly.
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    for key, entry in manifest["entries"].items():
        path = dest / entry["file"]
        assert _sha256_file(path) == entry["sha256"], key


@pytest.mark.unit
def test_exclusive_json_write_refuses_preexisting_file(tmp_path):
    """Payload JSON paths use O_EXCL — cannot clobber a pre-planted file."""
    from ambr.results import _write_bytes_exclusive, RunResultsIOError

    dest = tmp_path / "x.json"
    dest.write_text("planted\n", encoding="utf-8")
    with pytest.raises((FileExistsError, OSError, RunResultsIOError)):
        _write_bytes_exclusive(dest, b'{"ok": true}\n')
    assert dest.read_text(encoding="utf-8") == "planted\n"


@pytest.mark.unit
def test_manifest_tmp_symlink_cannot_escape_destination(tmp_path):
    """Pre-planted manifest.json.tmp symlink must not redirect writes outside root."""
    dest = tmp_path / "run"
    dest.mkdir()
    outside = tmp_path / "escaped.txt"
    outside.write_text("ORIGINAL_SECRET\n", encoding="utf-8")

    # Plant the *legacy* predictable temp name as a symlink to an external file.
    planted = dest / "manifest.json.tmp"
    planted.symlink_to(outside)

    r = RunResults({"info": {"v": 1}, "model": pl.DataFrame({"t": [1, 2]})})
    r.save(dest)

    # External target must be untouched
    assert outside.read_text(encoding="utf-8") == "ORIGINAL_SECRET\n"
    # Committed manifest is a regular file inside dest
    final = dest / "manifest.json"
    assert final.is_file()
    assert not final.is_symlink()
    manifest = json.loads(final.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    # Load works
    loaded = RunResults.load(dest)
    assert loaded["info"] == {"v": 1}
    assert loaded.model.height == 2


@pytest.mark.unit
def test_allow_fallback_false_requires_preferred_format(tmp_path, monkeypatch):
    r = RunResults({"model": pl.DataFrame({"t": [1]})})
    dest = tmp_path / "nofallback"

    def boom(self, path):  # noqa: ARG001
        raise RuntimeError("parquet unavailable")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises((RunResultsIOError, RuntimeError)):
        r.save(dest, format="parquet", allow_fallback=False)


@pytest.mark.unit
def test_experiment_canonical_and_legacy_kwargs():
    sample = am.Sample(
        {"steps": 3, "seed": [0, 1], "show_progress": False},
        n=2,
    )
    exp = am.Experiment(model_type=_Tiny, sample=sample, iterations=1)
    out = exp.run()
    assert "model" in out and out["model"].height >= 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp2 = am.Experiment(
            model_class=_Tiny,
            parameters=sample,
            iterations=1,
        )
        assert any("model_class" in str(x.message) or "deprecated" in str(x.message).lower() for x in w)
    out2 = exp2.run()
    assert out2["parameters"].height == 2


@pytest.mark.unit
def test_experiment_requires_sample_instance():
    with pytest.raises(TypeError, match="model_type"):
        am.Experiment(sample=am.Sample({"steps": 1}, n=1))
    with pytest.raises(TypeError, match="sample"):
        am.Experiment(model_type=_Tiny)
