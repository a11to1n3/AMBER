API Reference
=============

This section provides detailed documentation for all AMBER classes, functions, and modules.

Core Components
---------------

.. toctree::
   :maxdepth: 2

   model
   agent
   population
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

* :class:`ambr.Model` - Base class for all agent-based models
* :class:`ambr.Agent` - Individual agent with behaviors and properties
* :class:`ambr.BaseAgent` - Abstract base class for custom agents
* :class:`ambr.BaseModel` - Abstract base class for custom models
* :class:`ambr.Population` - SoA memory manager for high-performance state handling

**Environments:**

* :class:`ambr.GridEnvironment` - 2D grid-based spatial environment
* :class:`ambr.SpaceEnvironment` - Continuous 2D space environment  
* :class:`ambr.NetworkEnvironment` - Graph/network-based environment

**Data Structures:**

* :class:`ambr.AgentList` - List-like container for agents with additional functionality

**Optimization:**

* :func:`ambr.grid_search` - Exhaustive parameter space search
* :func:`ambr.random_search` - Random parameter sampling
* :func:`ambr.bayesian_optimization` - Bayesian optimization of model parameters

**Experiments:**

* :class:`ambr.Experiment` - Framework for running multiple model configurations
* :class:`ambr.Sample` - Parameter sampling for experiments
* :class:`ambr.IntRange` - Integer range specification for parameters 