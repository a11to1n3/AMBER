"""Deprecation helpers for the AMBER 0.4 -> 1.0 API transition.

New canonical APIs land in 0.4 alongside the old ones; the old ones keep working
but emit ``DeprecationWarning`` pointing at the replacement, and are removed in
1.0. Warnings are suppressed when the environment variable
``AMBER_SUPPRESS_DEPRECATIONS`` is set (so benchmark / reproducibility runs stay
quiet).

Use :func:`warn_deprecated` inside a deprecated method/property body, or the
:func:`deprecated` decorator on a deprecated function.
"""

import functools
import os
import warnings


def _suppressed() -> bool:
    return os.environ.get("AMBER_SUPPRESS_DEPRECATIONS", "") not in ("", "0", "false", "False")


def warn_deprecated(
    what: str,
    replacement: str,
    *,
    since: str = "0.4",
    remove_in: str = "1.0",
    stacklevel: int = 3,
) -> None:
    """Emit a one-line DeprecationWarning unless suppressed via env var."""
    if _suppressed():
        return
    warnings.warn(
        f"{what} is deprecated since AMBER {since} and will be removed in "
        f"{remove_in}; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(replacement: str, *, since: str = "0.4", remove_in: str = "1.0"):
    """Decorator marking a function/method as deprecated in favour of ``replacement``."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(
                func.__qualname__ + "()", replacement,
                since=since, remove_in=remove_in, stacklevel=3,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator
