Installation
============

Requirements
------------

AMBER requires Python 3.10 or higher and the following dependencies:

* **polars** >= 0.20.0 - High-performance DataFrame library
* **numpy** >= 1.21.0 - Numerical computing
* **networkx** >= 2.5 - Graph and network analysis

Optional plotting (``ambr[viz]``) pulls in **matplotlib** >= 3.3.0.
CI and documentation builds should set ``MPLBACKEND=Agg`` when a
non-interactive backend is required; AMBER never forces a matplotlib backend.

Install from PyPI
------------------

The easiest way to install AMBER is using pip:

.. code-block:: bash

   pip install ambr

This will install AMBER and all required dependencies.

Install from Source
-------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/a11to1n3/AMBER.git
   cd AMBER
   pip install -e .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/a11to1n3/AMBER.git
   cd AMBER
   pip install -e ".[dev]"

This includes testing, documentation, and code quality tools.

Verify Installation
-------------------

To verify that AMBER is installed correctly, run:

.. code-block:: python

   import ambr as am
   print(am.__version__)

You should see the version number printed without any errors.

Optional Dependencies
---------------------

AMBER extras (install what you need):

.. code-block:: bash

   # CPU acceleration (Numba + SciPy) — recommended on Mac / no-CUDA machines
   pip install 'ambr[perf]'

   # NVIDIA GPU lane (CuPy). Prefer a CUDA-matched wheel if the default fails,
   # e.g. ``pip install cupy-cuda12x`` instead of the generic ``cupy`` pin.
   # Apple Metal/MPS is **not** used.
   pip install 'ambr[gpu]'

   # SMAC Bayesian / multi-objective optimization (pins scikit-learn for SMAC 2.4)
   pip install 'ambr[advanced]'

   # Plot helpers (matplotlib; import ambr never pulls it until plot_* is used)
   pip install 'ambr[viz]'

   # Interactive example notebooks
   pip install 'ambr[examples]'

   # Documentation build tools
   pip install 'ambr[docs]'

Other useful packages:

* **jupyter** - interactive development
* **plotly** - interactive visualizations

.. code-block:: bash

   pip install jupyter plotly tqdm

Verify lanes after install::

   import ambr as am
   print(am.__version__)
   am.print_status()                 # GPU? Numba?
   print(am.recommend(10_000))
   # GPU claim samples (on an NVIDIA host with CuPy):
   #   python scripts/run_gpu_claims.py --quick

See :doc:`going_faster` for lanes, :doc:`reproducibility` for seed/device
policy, and :doc:`paper_and_package` for paper vs package numbers.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: If you get import errors, make sure all dependencies are installed:

.. code-block:: bash

   pip install --upgrade polars numpy networkx
   # optional plotting:
   # pip install 'ambr[viz]'

**Performance Issues**: For large simulations, consider:

* Using the latest version of Polars
* ``pip install 'ambr[perf]'`` for Numba-accelerated ``scatter_add`` / subset writes
* Installing numpy with accelerated BLAS libraries
* Running on systems with sufficient RAM
* For large-N runs on NVIDIA GPUs: ``pip install 'ambr[gpu]'`` (or a
  CUDA-matched CuPy wheel), then the same view-API model via
  ``model.gpu().run()``, or ``ArrayKernelModel`` — see :doc:`going_faster`

**SMAC / advanced install**: If SMAC fails with a scikit-learn ``DTYPE`` error,
reinstall the pinned extra::

   pip install 'ambr[advanced]'

**Jupyter Setup**: For interactive development with Jupyter:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/a11to1n3/AMBER/issues>`_
2. Read the documentation thoroughly
3. Ask questions in the community forums
4. Report bugs with minimal reproducible examples
