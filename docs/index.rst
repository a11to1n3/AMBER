AMBER Documentation
===================

**AMBER** (Agent-Based Modeling Environment for Research) is a powerful Python framework for building and running agent-based models. It provides a comprehensive toolkit for researchers and practitioners to create complex simulations with ease.

.. image:: https://img.shields.io/github/workflow/status/amber-team/amber/CI
   :target: https://github.com/amber-team/amber/actions
   :alt: CI Status

.. image:: https://codecov.io/gh/amber-team/amber/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/amber-team/amber
   :alt: Coverage

.. image:: https://img.shields.io/pypi/v/amber.svg
   :target: https://pypi.org/project/amber/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/amber.svg
   :target: https://pypi.org/project/amber/
   :alt: Python Versions

Features
--------

* **Intuitive API**: Simple and clean interface for building agent-based models
* **High Performance**: Efficient data structures using Polars for fast simulations
* **Flexible Environments**: Support for grid, continuous space, and network topologies
* **Rich Analytics**: Built-in data collection and analysis tools
* **Scalable**: Handles models from small prototypes to large-scale simulations
* **Extensible**: Easy to extend with custom agent behaviors and environments

Quick Start
-----------

Install AMBER using pip:

.. code-block:: bash

   pip install amber

Create your first model:

.. code-block:: python

   import amber as am
   
   class SimpleModel(am.Model):
       def setup(self):
           # Create 100 agents
           for i in range(100):
               agent = am.Agent(self, i)
               self.add_agent(agent)
       
       def step(self):
           pass  # Define agent behaviors here
   
   # Run the model
   model = SimpleModel({'steps': 50})
   results = model.run()

For more examples, check the ``examples/`` directory in the repository.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 