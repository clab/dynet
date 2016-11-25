.. _operations:

Operations
==========

Operation Interface
-------------------

The following functions define DyNet "Expressions," which are used as an interface to
the various functions that can be used to build DyNet computation graphs. Expressions
for each specific function are listed below.

.. doxygengroup:: operations
	:members:
	:content-only:

Input Operations
----------------

These operations allow you to input something into the computation graph, either simple
scalar/vector/matrix inputs from floats, or parameter inputs from a DyNet parameter
object. They all requre passing a computation graph as input so you know which graph
is being used for this particular calculation.

.. doxygengroup:: inputoperations
	:members:
	:content-only:

Arithmetic Operations
---------------------

These operations perform basic arithemetic over values in the graph.

.. doxygengroup:: arithmeticoperations
	:members:
	:content-only:

Probability/Loss Operations
---------------------------

These operations are used for calculating probabilities, or calculating loss functions
for use in training.

.. doxygengroup:: lossoperations
	:members:
	:content-only:

Flow/Shaping Operations
-----------------------

These operations control the flow of information through the graph, or the shape of
the vectors/tensors used in the graph.

.. doxygengroup:: flowoperations
	:members:
	:content-only:

Noise Operations
----------------

These operations are used to add noise to the graph for purposes of making learning
more robust.

.. doxygengroup:: noiseoperations
	:members:
	:content-only:

Tensor Operations
-----------------

These operations are used for performing operations on higher order tensors.

.. doxygengroup:: tensoroperations
	:members:
	:content-only:

Linera Algebra Operations
-------------------------

These operations are used for performing various operations common in linear algebra.

.. doxygengroup:: linalgoperations
	:members:
	:content-only:
