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


def test_bayesian_optimization_docs_do_not_claim_gaussian_process():
    from ambr.optimization import bayesian_optimization

    doc = bayesian_optimization.__doc__ or ""
    assert "RandomForest" in doc
    assert "Gaussian Process facade" not in doc
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "SMAC Gaussian-process" not in readme
    assert "RandomForest" in readme
    assert "ambr[gpu]" in readme
    assert "ambr[viz]" in readme
    assert "print(results.model)" not in readme
    assert "print(results.agents.head())" not in readme


def test_contact_email_is_not_a_placeholder():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "example.com" not in ambr.__email__
    assert "@uni-wuerzburg.de" in ambr.__email__
    assert "example.com" not in pyproject
    assert ambr.__email__ in pyproject
    assert ambr.__email__ in citation
    assert "example.com" not in citation


def test_citation_release_date_matches_changelog():
    """CITATION.cff, CHANGELOG.md, and docs/changelog.rst must agree on 0.5.0."""
    import re

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    rst = (ROOT / "docs" / "changelog.rst").read_text(encoding="utf-8")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    proj_ver = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', pyproject, re.M)
    cite_date = re.search(r'^date-released:\s*"(\d{4}-\d{2}-\d{2})"', citation, re.M)
    cite_ver = re.search(r'^version:\s*"([^"]+)"', citation, re.M)
    md_ver = re.search(r"^## v(\d+\.\d+\.\d+) - (\d{4}-\d{2}-\d{2})", changelog, re.M)
    rst_ver = re.search(r"^\[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})", rst, re.M)

    assert proj_ver and cite_date and cite_ver and md_ver and rst_ver
    assert cite_ver.group(1) == md_ver.group(1) == rst_ver.group(1) == proj_ver.group(1)
    assert cite_date.group(1) == md_ver.group(2) == rst_ver.group(2)


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
