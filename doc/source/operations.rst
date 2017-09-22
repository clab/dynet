.. _operations:

Operations
----------

Operation Interface
~~~~~~~~~~~~~~~~~~~

The following functions define DyNet "Expressions," which are used as an interface to
the various functions that can be used to build DyNet computation graphs. Expressions
for each specific function are listed below.

.. doxygengroup:: operations
	:members:
	:content-only:

Input Operations
~~~~~~~~~~~~~~~~

These operations allow you to input something into the computation graph, either simple
scalar/vector/matrix inputs from floats, or parameter inputs from a DyNet parameter
object. They all requre passing a computation graph as input so you know which graph
is being used for this particular calculation.

.. doxygengroup:: inputoperations
	:members:
	:content-only:

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

These operations perform basic arithemetic over values in the graph.

.. doxygengroup:: arithmeticoperations
	:members:
	:content-only:

Probability/Loss Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These operations are used for calculating probabilities, or calculating loss functions
for use in training.

.. doxygengroup:: lossoperations
	:members:
	:content-only:

Flow/Shaping Operations
~~~~~~~~~~~~~~~~~~~~~~~

These operations control the flow of information through the graph, or the shape of
the vectors/tensors used in the graph.

.. doxygengroup:: flowoperations
	:members:
	:content-only:

Noise Operations
~~~~~~~~~~~~~~~~

These operations are used to add noise to the graph for purposes of making learning
more robust.

.. doxygengroup:: noiseoperations
	:members:
	:content-only:

Tensor Operations
~~~~~~~~~~~~~~~~~

These operations are used for performing operations on higher order tensors.

**Remark**: Compiling the contraction operations takes a lot of time with CUDA. For this reason, only the CPU implementation is compiled by default. If you need those operations, you need to un-comment `this line <https://github.com/clab/dynet/blob/master/dynet/nodes-contract.cc#L11>`_ in the source before compiling. TODO: make this simpler.

.. doxygengroup:: tensoroperations
	:members:
	:content-only:

Linear Algebra Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

These operations are used for performing various operations common in linear algebra.

.. doxygengroup:: linalgoperations
	:members:
	:content-only:

Convolution Operations
~~~~~~~~~~~~~~~~~~~~~~

These operations are convolution-related.

.. doxygengroup:: convolutionoperations
	:members:
	:content-only:

Normalization Operations
~~~~~~~~~~~~~~~~~~~~~~~~

This includes batch normalization and the likes.

.. doxygengroup:: normoperations
	:members:
	:content-only:

Device operations
~~~~~~~~~~~~~~~~

These operations are device-related.

.. doxygengroup:: deviceoperations
  :members:
  :content-only:
