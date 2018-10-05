Core functionalities
--------------------

Computation Graph
~~~~~~~~~~~~~~~~~

The ComputationGraph is the workhorse of DyNet. From the `DyNet technical report <https://arxiv.org/abs/1701.03980>`_ :

    [The] computation graph represents symbolic computation, and the results of the computation are evaluated lazily: the computation is only performed once the user explicitly asks for it (at which point a “forward” computation is triggered).
    Expressions that evaluate to scalars (i.e. loss values) can also be used to trigger a “backward” computation, computing the gradients of the computation with respect to the parameters.

.. doxygengroup:: compgraph
    :members:
    :content-only:

Nodes
~~~~~

Nodes are constituents of the computation graph. The end user doesn't interact with Nodes but with Expressions.

However implementing new operations requires to create a new subclass of the Node class described below.

.. doxygengroup:: nodes
    :members:
    :content-only:

Parameters and Model
~~~~~~~~~~~~~~~~~~~~

Parameters are things that are optimized. in contrast to a system like Torch where computational modules may have their own parameters, in DyNet parameters are just parameters.

To deal with sparse updates, there are two parameter classes:

- Parameters represents a vector, matrix, (eventually higher order tensors)
  of parameters. These are densely updated.
- LookupParameters represents a table of vectors that are used to embed a
  set of discrete objects. These are sparsely updated.

.. doxygengroup:: params
    :members:
    :content-only:

Tensor
~~~~~~

Tensor objects provide a bridge between C++ data structures and Eigen Tensors for multidimensional data.

Concretely, as an end user you will obtain a tensor object after calling ``.value()`` on an expression. You can then use functions described below to convert these tensors to ``float`` s, arrays of ``float`` s, to save and load the values, etc...

Conversely, when implementing low level nodes (e.g. for new operations), you will need to retrieve Eigen tensors from DyNet tensors in order to perform efficient computation.

.. doxygengroup:: tensor
    :members:
    :content-only:

Dimensions
~~~~~~~~~~

The Dim class holds information on the shape of a tensor. As explained in :doc:`unorthodox`, in DyNet the dimensions are represented as the standard dimension + the batch dimension, which makes batched computation transparent.

.. doxygengroup:: dim
    :members:
    :content-only:
