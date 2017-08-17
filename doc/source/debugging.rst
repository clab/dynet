.. _debugging:

Debugging/Reporting Issues
==========================

There are a number of tools to make debugging easier in DyNet.
In addition, we welcome any questions or issues, and will be able to respond most effectively if you follow the guidelines below.

Debugging Tools
---------------

Visualization
~~~~~~~~~~~~~

It is possible to create visualizations of the computation graph by calling the ``print_graphviz()`` function, which can be helpful to debug. When this functionality is used in Python, it is necessary to add the command line argument ``--dynet-viz``. In Python, there is also a ``print_text_graphviz()`` function which  will be less pretty than the ``print_graphviz()`` function, but doesn't require the command line flag.

Immediate Computation
~~~~~~~~~~~~~~~~~~~~~

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


.. _debugging-asking:

Asking Questions/Reporting Bugs
-------------------------------

Feel free to contact the `dynet-users <https://groups.google.com/forum/#!forum/dynet-users>`_ group or file an issue on `github <https://github.com/clab/dynet>`_ with any questions or problems. 
(If you subscribe to ``dynet-users`` and want to receive email make sure to select "all email" when you sign up.)

When you have an issue, including the following information in your report will greatly help us debug:

* What is the error? Copy and paste the error message.
* What is your environment? Are you running on CPU or GPU? What OS? If the problem seems to be related to a specific library (CUDA, Eigen), what version of that library are you using?
* If possible, it will be really really helpful if you can provide a minimal code example that will cause the problem to occur. This way the developers will be able to reproduce the problem in their own environment.

If you have a build problem and want to debug, please run

::

    make clean
    make VERBOSE=1 &> make.log

then examine the commands in the ``make.log`` file to see if anything
looks fishy. If you would like help, send this ``make.log`` file via the
"Issues" tab on GitHub, or to the dynet-users mailing list.
