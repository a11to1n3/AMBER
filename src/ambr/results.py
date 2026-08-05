"""Run result containers with AgentPy-friendly attribute access.

``Model.run()`` returns a :class:`RunResults` mapping. You can use either::

    results['agents']      # dict-style (stable, existing code)
    results.agents         # attribute-style (AgentPy / notebook friendly)

Both forms return the same objects.

Persistence
-----------
::

    results.save("out/run_001")
    restored = RunResults.load("out/run_001")

Writes ``info.json`` plus Polars parquet for ``model`` / ``agents`` frames
when present. Extra keys with Polars DataFrames are also written as
``{key}.parquet``; other JSON-serializable values go into ``extra.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


class RunResults(dict):
    """Dict of run artifacts (``info``, ``agents``, ``model``, …) with attr access.

    Subclasses :class:`dict`, so ``results['agents']``, ``'model' in results``,
    and ``json``/``**results`` keep working. Attribute access is sugar for the
    same keys (inspired by AgentPy's ``DataDict``).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"{type(self).__name__!r} has no key {name!r}; "
                f"available: {sorted(self)}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep normal dict keys as mapping items, not instance attrs.
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self))
        return f"<RunResults keys=[{keys}]>"

    # --- AgentPy DataDict-ish helpers ---------------------------------------

    def keys_overview(self) -> Dict[str, str]:
        """Return a short map of key → type summary (notebook / debugging)."""
        out: Dict[str, str] = {}
        for k, v in self.items():
            if hasattr(v, "shape") and hasattr(v, "columns"):
                out[k] = f"DataFrame shape={getattr(v, 'shape', '?')} cols={list(v.columns)}"
            elif isinstance(v, dict):
                out[k] = f"dict keys={sorted(v)}"
            elif isinstance(v, list):
                out[k] = f"list len={len(v)}"
            else:
                out[k] = type(v).__name__
        return out

    def model_frame(self):
        """Return the model-level Polars frame (``results['model']``)."""
        return self["model"]

    def agents_frame(self):
        """Return the end-of-run agents Polars frame (``results['agents']``)."""
        return self["agents"]

    def save(self, path: PathLike) -> Path:
        """Persist this run to a directory.

        Layout::

            path/
              info.json          # if present
              model.parquet      # if present
              agents.parquet     # if present
              {key}.parquet      # other Polars DataFrames
              extra.json         # remaining JSON-serializable values

        Returns the directory path.
        """
        import polars as pl

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        extra: Dict[str, Any] = {}
        for key, value in self.items():
            if key == "info" and isinstance(value, dict):
                (root / "info.json").write_text(
                    json.dumps(value, default=str, indent=2) + "\n"
                )
                continue
            if isinstance(value, pl.DataFrame):
                value.write_parquet(root / f"{key}.parquet")
                continue
            if key == "contract" and isinstance(value, list):
                # Certificates: store a lightweight summary only
                summary = []
                for cert in value:
                    summary.append(
                        {
                            "step": getattr(cert, "step", None),
                            "ok": getattr(cert, "ok", None),
                            "clean": getattr(cert, "clean", None),
                            "n_violations": len(getattr(cert, "violations", []) or []),
                        }
                    )
                (root / "contract_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                continue
            try:
                json.dumps(value, default=str)
                extra[key] = value
            except TypeError:
                extra[key] = repr(value)
        if extra:
            (root / "extra.json").write_text(
                json.dumps(extra, default=str, indent=2) + "\n"
            )
        return root

    @classmethod
    def load(cls, path: PathLike) -> "RunResults":
        """Load a directory written by :meth:`save`."""
        import polars as pl

        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"RunResults directory not found: {root}")
        data: Dict[str, Any] = {}
        info_path = root / "info.json"
        if info_path.is_file():
            data["info"] = json.loads(info_path.read_text())
        for pq in sorted(root.glob("*.parquet")):
            data[pq.stem] = pl.read_parquet(pq)
        extra_path = root / "extra.json"
        if extra_path.is_file():
            extra = json.loads(extra_path.read_text())
            if isinstance(extra, dict):
                for k, v in extra.items():
                    data.setdefault(k, v)
        contract_path = root / "contract_summary.json"
        if contract_path.is_file() and "contract" not in data:
            data["contract_summary"] = json.loads(contract_path.read_text())
        return cls(data)
