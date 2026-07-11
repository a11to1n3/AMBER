"""Run result containers with AgentPy-friendly attribute access.

``Model.run()`` returns a :class:`RunResults` mapping. You can use either::

    results['agents']      # dict-style (stable, existing code)
    results.agents         # attribute-style (AgentPy / notebook friendly)

Both forms return the same objects.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Optional


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
