"""Id → row-position index for columnar writes.

Single source of truth for mapping agent ids to DataFrame row positions.
Used by:

* the **vectorized view API** (``sequences``) — subset assigns / ``scatter_add``
* the **OOP write flush** (``model._flush_pending_writes``)

Caches live on the model instance and are invalidated by
``Model._bump_id_version`` whenever the id set changes:

* ``_ids_arange_cache`` — ``(id_version, bool)`` whether ids are exactly ``0..N-1``
* ``_id_pos_cache`` — ``(id_version, {id: row})`` hash map for the general case

Do not re-implement this lookup elsewhere; import :func:`resolve_positions`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl


def ids_are_arange(model: Any, df_ids_np: np.ndarray) -> bool:
    """Return True if ``df_ids_np`` is exactly ``0, 1, ..., N-1``.

    Result is cached on ``model`` per ``_id_version`` so scatter/flush do not
    re-scan the full id column every call within a stable population.
    """
    version = getattr(model, "_id_version", 0)
    cached = getattr(model, "_ids_arange_cache", None)
    if cached is not None and cached[0] == version:
        return cached[1]

    n = int(df_ids_np.size)
    ok = (
        df_ids_np.dtype.kind in ("i", "u")
        and n > 0
        and int(df_ids_np[0]) == 0
        and int(df_ids_np[-1]) == n - 1
        and (
            n == 1
            or bool(
                np.all(
                    df_ids_np.astype(np.int64, copy=False)
                    == np.arange(n, dtype=np.int64)
                )
            )
        )
    )
    model._ids_arange_cache = (version, ok)
    return ok


def resolve_positions(model: Any, df: pl.DataFrame, ids_np: np.ndarray) -> np.ndarray:
    """Map agent ids to row positions in ``df``.

    Fast path: when population ids are ``0..N-1`` (typical after
    ``add_agents``), positions equal the ids themselves — O(1) beyond the
    one-time arange check.

    Slow path: hash map ``{id: row}`` cached on the model until the id set
    changes (``_bump_id_version``).

    Returns
    -------
    np.ndarray
        int64 row indices aligned with ``ids_np`` (may contain duplicates when
        the view is a scatter list).
    """
    df_ids_np = df["id"].to_numpy()

    # Contiguous [0, N) — common after bulk create.
    if df_ids_np.size == df.height and ids_are_arange(model, df_ids_np):
        return np.asarray(ids_np, dtype=np.int64)

    # General id → row lookup, reused within the same id-version.
    cached: Optional[Tuple[int, Dict[int, int]]] = getattr(model, "_id_pos_cache", None)
    version = getattr(model, "_id_version", 0)
    if cached is None or cached[0] != version:
        id_to_pos = {int(v): i for i, v in enumerate(df_ids_np)}
        model._id_pos_cache = (version, id_to_pos)
    else:
        id_to_pos = cached[1]

    return np.fromiter(
        (id_to_pos[int(v)] for v in ids_np),
        dtype=np.int64,
        count=int(ids_np.size),
    )
