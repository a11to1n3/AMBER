Changelog
=========

All notable changes to AMBER will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.1.2] - 2026-01-28
---------------------

Added
~~~~~
- ``AttrDict`` class for AgentPy-compatible attribute-style parameter access (``self.p.param_name``)
- Method forwarding on ``AgentList`` for AgentPy-style syntax (``agents.method()`` calls method on all agents)

Changed
~~~~~~~
- ``AgentList.call()`` now returns list of results from each agent's method call
- ``BaseModel`` and ``BaseAgent`` now use ``AttrDict`` for parameter storage

[0.1.1] - 2026-01-21
---------------------

Fixed
~~~~~
- Renamed stale `amber` imports to `ambr` in example notebooks and documentation (`# <PR/Commit>`)
- Corrected version consistency across `setup.py`, `__init__.py`, and documentation

Changed
~~~~~~~
- Bumped version to 0.1.1
Added
~~~~~
- Comprehensive documentation with ReadTheDocs integration
- Jupyter notebook examples converted to Python scripts
- Tutorial and API reference documentation
- Contributing guidelines and development workflow

[0.1.0] - 2024-06-19
---------------------

Added
~~~~~
- Initial release of AMBER framework
- Core Model and Agent classes
- GridEnvironment, SpaceEnvironment, and NetworkEnvironment
- AgentList for managing agent collections
- Optimization module with grid search, random search, and Bayesian optimization
- Experiment framework for parameter sweeps
- Comprehensive test suite with 100% pass rate
- GitHub Actions CI/CD pipeline
- Coverage reporting with Codecov
- Example notebooks for various simulation types

Features
~~~~~~~~
- **High-Performance Data Structures**: Using Polars for fast DataFrame operations
- **Flexible Environments**: Support for grid, continuous space, and network topologies
- **Built-in Optimization**: Parameter tuning with multiple optimization algorithms
- **Comprehensive Testing**: Full test coverage across all modules
- **Professional Documentation**: Sphinx-based documentation with examples

Fixed
~~~~~
- Model execution flow and data recording issues
- Environment class compatibility with different input formats
- Random number generation compatibility across numpy versions
- DataFrame schema consistency in agent data management
- Mock object compatibility in test environments

Changed
~~~~~~~
- Moved from examples/ to docs/examples/ for better organization
- Enhanced README with professional presentation
- Improved error handling and user feedback

Technical Details
~~~~~~~~~~~~~~~~~
- **Dependencies**: Polars, NumPy, NetworkX, Matplotlib, Seaborn
- **Python Support**: 3.8+
- **Testing**: pytest with comprehensive coverage
- **Documentation**: Sphinx with ReadTheDocs theme
- **CI/CD**: GitHub Actions with multi-OS and multi-Python testing 