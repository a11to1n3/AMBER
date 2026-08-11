"""Run provenance helpers for ``results.info``.

Captures enough metadata to reproduce and audit a single ``Model.run()``
(or array-kernel) outcome without relying on ambient notebook state.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


def _iso_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cupy_cuda_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"cupy_version": None, "cuda_version": None}
    try:
        import cupy as cp  # type: ignore

        out["cupy_version"] = getattr(cp, "__version__", None)
        try:
            # e.g. (12, 0) → "12.0"
            ver = cp.cuda.runtime.runtimeGetVersion()
            major, minor = divmod(int(ver), 1000)
            out["cuda_version"] = f"{major}.{minor}"
        except Exception:
            out["cuda_version"] = None
    except Exception:
        pass
    return out


def _git_revision() -> Optional[str]:
    """Best-effort git SHA of the application / checkout.

    Order: ``AMBER_GIT_REVISION`` env, then ``git rev-parse HEAD`` in CWD.
    """
    env = os.environ.get("AMBER_GIT_REVISION") or os.environ.get("AMBER_APP_REVISION")
    if env:
        return env.strip()
    try:
        import subprocess

        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return sha or None
    except Exception:
        return None


def config_hash(model_class: str, parameters: Mapping[str, Any], seed: Any) -> str:
    """Stable short hash of model class + parameters + seed."""
    payload = {
        "model_class": model_class,
        "parameters": dict(parameters) if parameters is not None else {},
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def qualified_class_name(obj_or_cls: Any) -> str:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return f"{cls.__module__}.{cls.__qualname__}"


def build_run_info(
    model: Any,
    *,
    steps: int,
    start_time: float,
    end_time: float,
    device: str = "cpu",
    mode: str = "vectorized",
    status: str = "completed",
    run_uuid: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a rich ``results.info`` provenance record.

    Parameters
    ----------
    model:
        Model instance after the run (reads ``p``, class, optional attrs).
    steps:
        Steps completed (typically ``model.t``).
    start_time / end_time:
        ``time.time()`` timestamps bracketing the run.
    device / mode:
        Execution lane (``cpu``/``gpu``, ``vectorized``/``oop``).
    status:
        ``completed``, ``failed``, ``interrupted``, etc.
    run_uuid:
        Optional pre-allocated UUID; generated if omitted.
    extra:
        Optional caller-supplied fields merged last.
    """
    try:
        import ambr as am

        ambr_ver = getattr(am, "__version__", None)
    except Exception:
        ambr_ver = _pkg_version("ambr")

    params = {}
    try:
        params = dict(getattr(model, "p", {}) or {})
    except Exception:
        params = {}

    seed = params.get("seed", getattr(model, "seed", None))
    model_cls = qualified_class_name(model)
    rid = run_uuid or str(uuid.uuid4())
    cupy_cuda = _cupy_cuda_versions()

    info: Dict[str, Any] = {
        # Core run identity
        "run_uuid": rid,
        "status": status,
        "completion_status": status,
        "steps": steps,
        "run_time": float(end_time - start_time),
        "start_time": _iso_utc(start_time),
        "end_time": _iso_utc(end_time),
        # Model + configuration
        "model_class": model_cls,
        "parameters": params,
        "seed": seed,
        "config_hash": config_hash(model_cls, params, seed),
        # Execution lane
        "device": device,
        "mode": mode,
        "execution_lane": f"{device}/{mode}",
        # Software stack
        "ambr_version": ambr_ver,
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "polars_version": _pkg_version("polars"),
        "numpy_version": _pkg_version("numpy"),
        "cupy_version": cupy_cuda["cupy_version"],
        "cuda_version": cupy_cuda["cuda_version"],
        # Optional application / checkout revision
        "git_revision": _git_revision(),
        "application_revision": os.environ.get("AMBER_APP_REVISION"),
    }
    if extra:
        info.update(dict(extra))
    return info
