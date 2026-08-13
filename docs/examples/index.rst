Examples
========

This section describes the comprehensive examples demonstrating various features and use cases of AMBER. All examples are available as Python scripts in the ``examples/`` directory for direct execution.

Getting Started Examples
-------------------------

**Wealth transfer (quickstart / README)**
   Canonical dual-lane wealth model lives in the package README and
   :doc:`../quickstart` (OOP + vectorized view API). Copy those samples
   rather than a separate ``wealth_transfer.py`` script.

**Schelling (vectorized grid)**
   Canonical occupancy helpers + view-API Schelling (see
   :doc:`../environments_schelling`).

   - Script: ``examples/schelling_vectorized.py``

**GPU quickstart**
   Native placement (``model.gpu().run()`` when NVIDIA+CuPy is available)
   on a vectorized view-API model with ``step_vectorized``, plus
   ``ArrayKernelModel``. See :doc:`../going_faster`.

   - Script: ``examples/gpu_quickstart.py``
   - Host verification: ``python scripts/run_gpu_claims.py --quick``

**Segregation Model**
   Schelling-style segregation with optional plotting.

   - Script: ``examples/segregation_model.py``

Advanced Examples
-----------------

**Virus Spread Simulation**
   Epidemiological model simulating disease spread through a population.

   - Script: ``examples/virus_spread_simulation.py``

**Forest Fire Model**
   Cellular automaton model of wildfire spread.

   - Script: ``examples/forest_fire_simulation.py``
   - Notebook: ``examples/forest_fire_simulation.ipynb`` (regenerated from the script)

**Flocking Simulation**
   Boids-style flocking (see also ``examples/flocking_tensor.py``).

   - Script: ``examples/flocking_simulation.py``
   - Notebook: ``examples/flocking_simulation.ipynb`` (regenerated from the script)

**Button Network Simulation**
   Network percolation (Kauffman buttons + threads).

   - Script: ``examples/button_network_simulation.py``
   - Notebook: ``examples/button_network_simulation.ipynb`` (regenerated from the script)

Parameter Optimization & Calibration
-------------------------------------

**Ensemble / SMAC smoke**
   GPU/CPU ensemble always; SMAC path skipped honestly without
   ``ambr[advanced]``.

   - Script: ``examples/smac_batch_sir_smoke.py``

**Simple SMAC Calibration**
   Introduction to SMAC optimization with AMBER.

   - Script: ``examples/smac_calibration_simple.py``

**Comprehensive SMAC Calibration**
   Single-objective SMACOptimizer workflows.

   - Script: ``examples/smac_calibration_basic.py``

**Multi-Objective SMAC Optimization**
   MultiObjectiveSMAC / Pareto-style examples.

   - Script: ``examples/smac_calibration_advanced.py``

Running the Examples
--------------------

**Python Scripts**

All listed scripts live under ``examples/``:

.. code-block:: bash

   cd examples
   python gpu_quickstart.py
   python schelling_vectorized.py
   python smac_batch_sir_smoke.py

**Requirements**

Some examples may require additional dependencies:

.. code-block:: bash

   pip install 'ambr[perf]'              # Numba CPU path
   pip install 'ambr[gpu]'               # NVIDIA + CuPy only (not Metal/MPS)
   pip install 'ambr[advanced]'          # SMAC search (no plots)
   pip install 'ambr[advanced,viz]'      # SMAC examples with matplotlib plots
   pip install 'ambr[viz,examples]'      # notebooks / interactive extras

**Example Structure**

Each Python script is self-contained and includes:

- Model definition and setup
- Simulation execution
- Data analysis and visualization
- Clear comments explaining the logic

Learning Path
-------------

We recommend following this sequence for learning AMBER:

1. **Start with Wealth Transfer** - Learn basic model structure and agent interactions
2. **Try Segregation Model** - Understand spatial environments and agent movement
3. **Explore Virus Spread** - See how to model state changes and interventions
4. **Advanced Models** - Forest fire, flocking, and network models for complex behaviors
5. **Interactive Examples** - Learn about real-time visualization and user interaction
6. **Parameter Optimization** - Automate parameter tuning with SMAC calibration

Each example builds on concepts from previous ones while introducing new features and techniques.

Source Code
-----------

All example source code can be found in the project repository under the ``examples/`` directory. The examples are designed to be:

- **Educational** - Clear, well-commented code that teaches AMBER concepts
- **Runnable** - Complete scripts that work out of the box
- **Extensible** - Easy to modify and build upon for your own projects
