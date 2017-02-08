Core functionalities
====================

Computation Graph
-----------------

Parameters and Model
--------------------

Parameters are things that are optimized. in contrast to a system like Torch where computational modules may have their own parameters, in DyNet parameters are just parameters.

To deal with sparse updates, there are two parameter classes:

- Parameters represents a vector, matrix, (eventually higher order tensors)
  of parameters. These are densely updated.
- LookupParameters represents a table of vectors that are used to embed a
  set of discrete objects. These are sparsely updated.

.. doxygengroup:: params
    :members:
    :content-only:

