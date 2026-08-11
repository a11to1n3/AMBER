"""Package metadata and release-surface checks."""

from importlib import util
from importlib.metadata import version
from pathlib import Path

import ambr


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_version_matches_distribution_metadata():
    assert ambr.__version__ == version("ambr")


def test_documentation_version_matches_distribution_metadata():
    spec = util.spec_from_file_location("ambr_docs_conf", ROOT / "docs" / "conf.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.release == version("ambr")
    assert module.version == version("ambr")


def test_python_support_floor_is_declared_consistently():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in pyproject
    assert '"Programming Language :: Python :: 3.10"' in pyproject
    assert '"Programming Language :: Python :: 3.9"' not in pyproject


def test_local_only_paths_are_excluded_from_release_surface():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "paper/" in gitignore
    assert "paper.zip" in gitignore
    assert ".claude/" in gitignore
    assert "prune .claude" in manifest
    assert "prune .github" in manifest
    assert "prune benchmarks" in manifest
    assert "prune docs" in manifest
    assert "prune examples" in manifest
    assert "prune paper" in manifest
    assert "prune tests" in manifest
    assert "exclude paper.zip" in manifest
