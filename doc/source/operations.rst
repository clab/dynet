DyNet Operations
================

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
