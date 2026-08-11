"""Optional build-time metadata.

Packaging / CI may overwrite ``GIT_REVISION`` with the AMBER repository SHA
at wheel build time. Editable installs leave it as ``\"unknown\"``; runtime
provenance then relies on the ``AMBER_GIT_REVISION`` environment variable
instead of probing the caller's working directory.
"""

from __future__ import annotations

# Set by release packaging when available. Do not read git from CWD at runtime.
GIT_REVISION = "unknown"
