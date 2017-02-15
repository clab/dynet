Debugging
==============
To make debugging simpler, you can use immediate computing mode in dynet. In this mode, every line of code gets executed immediately, just like imperative programming, so that you can find exactly where goes wrong. 

Further, dynet can automatically check validity of your model, i.e., detecting Inf/NaN, if it is in immediate computing mode. 

C++
--------------

You can switch to the immediate computing mode by calling ComputationGraph::set_immediate_compute as follows:

.. code:: cpp

    ComputationGraph cg;
    cg.set_immediate_compute(true);

To activate checking validity, you can add the following code after switching to immediate computing mode.

.. code:: cpp

    cg.set_check_validity(true);

Python
--------------
These features in python are in progress.
