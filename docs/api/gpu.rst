GPU backend
===========

An array-module abstraction over CuPy with a NumPy fallback, so device code is
portable: ``get_array_module``, ``to_device``, and ``to_host`` resolve to CuPy
when a GPU is present and fall back to NumPy when it is not.

.. automodule:: ambr.gpu
   :members:
   :undoc-members:
   :show-inheritance:
