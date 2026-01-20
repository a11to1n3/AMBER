API Reference
=============

This section provides detailed documentation for all AMBER classes, functions, and modules.

Core Components
---------------

.. toctree::
   :maxdepth: 2

   model
   agent
   environments
   sequences

Utilities
---------

.. toctree::
   :maxdepth: 2

   optimization
   experiment
   base

Quick Reference
---------------

**Core Classes:**

* :class:`amber.Model` - Base class for all agent-based models
* :class:`amber.Agent` - Individual agent with behaviors and properties
* :class:`amber.BaseAgent` - Abstract base class for custom agents
* :class:`amber.BaseModel` - Abstract base class for custom models

**Environments:**

* :class:`amber.GridEnvironment` - 2D grid-based spatial environment
* :class:`amber.SpaceEnvironment` - Continuous 2D space environment  
* :class:`amber.NetworkEnvironment` - Graph/network-based environment

**Data Structures:**

* :class:`amber.AgentList` - List-like container for agents with additional functionality

**Optimization:**

* :func:`amber.grid_search` - Exhaustive parameter space search
* :func:`amber.random_search` - Random parameter sampling
* :func:`amber.bayesian_optimization` - Bayesian optimization of model parameters

**Experiments:**

* :class:`amber.Experiment` - Framework for running multiple model configurations
* :class:`amber.Sample` - Parameter sampling for experiments
* :class:`amber.IntRange` - Integer range specification for parameters 