Debugging
==============

There are a number of tools to make debugging easier in DyNet.

Visualization
-------------

It is possible to create visualizations of the computation graph by calling the ``print_graphviz()`` function, which can be helpful to debug. When this functionality is used in Python, it is necessary to add the command line argument ``--dynet-viz``.

Immediate Computation
---------------------

In general, DyNet performs symbolic execution. This means that you first create the computation graph, then the computation will actually be performed when you request a value using functions such as ``forward()`` or ``value()``. However, if an error occurs during calculation, this can be hard to debug because the error doesn't occur immediately where the offending graph node is created. To make debugging simpler, you can use immediate computing mode in dynet. In this mode, every computation gets executed immediately, just like imperative programming, so that you can find exactly where goes wrong. 

In C++, you can switch to the immediate computing mode by calling ComputationGraph::set_immediate_compute as follows:

.. code:: cpp

    ComputationGraph cg;
    cg.set_immediate_compute(true);

Further, dynet can automatically check validity of your model, i.e., detecting Inf/NaN, if it is in immediate computing mode. To activate checking validity, you can add the following code after switching to immediate computing mode.

.. code:: cpp

    cg.set_check_validity(true);

In Python, these values can be set by using optional arguments to the ``renew_cg()`` function as follows:

.. code:: python

    dy.renew_cg(immediate_compute = True, check_validity = True)
