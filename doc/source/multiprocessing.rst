Multi-processing
================

In addition to minibatch support, the DyNet C++ API also supports training models using many CPU cores (Python support is pending).
This is particularly useful when performing training of networks that are not conducive to simple mini-batching, such as tree-structured networks.

DyNet abstracts most of the behind-the-scenes grit from the user.
The user defines a function to be called for each datum in the training data set, and passes this function, along with an array of data, to DyNet.
Internally, DyNet launches a pool of training processes and automatically handles passing data examples to each worker.
Each worker process individually processes a datum, computing the results of the forward and backward passes, computes gradients with respect to each parameter, and passes these results back to the parent process via a shared memory variable.
Whenever the parent process, which is also processing data, completes a gradient computation, it averages all of the gradients currently in the shared memory gradient storage and updates all parameters with respect to that average gradient.
In this way running training on ``n`` cores is similar to training with a stochastic minibatch size with expected value of approximately ``n``.
This method is quite efficient, achieving nearly linear speedups with increasing numbers of cores, due to its lockless nature.

Examples of how to use the multi-processing API can be found in the ``xor-mp`` and ``rnnlm-mp`` sections of the ``examples/cpp`` directory.
