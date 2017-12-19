.. _command-line-options:

Command Line Options
====================

All programs using DyNet have a few command line options. These must be
specified at the very beginning of the command line, before other
options.

-  ``--dynet-mem NUMBER``: DyNet runs by default with 512MB of memory,
   which is split evenly for the forward and backward steps, parameter
   storage as well as scratch use. This will be expanded automatically every
   time one of the pools runs out of memory. By setting NUMBER here, DyNet
   will allocate more memory immediately at the initialization stage.
   Note that you can also individually set the amount of memory for
   forward calculation, backward calculation, parameters, and scratch use by 
   using comma separated variables ``--dynet-mem FOR,BACK,PARAM,SCRATCH``. This is
   useful if, for example, you are performing testing and don't need to
   allocate any memory for backward calculation.
-  ``--dynet-weight-decay NUMBER``: Adds weight decay to the parameters,
   which modifies each parameter w such that `w *= (1-weight_decay)` after
   every update. This is similar to L2 regularization, but different in a
   couple ways, which are noted in detail in the "Unorthodox Design"
   section.
-  ``--dynet-autobatch NUMBER``: Turns on DyNet's automatic operation
   batching capability. This makes it possible to speed up computation with
   a minimum of work. More information about this functionality can be found
   `here <http://dynet.readthedocs.io/en/latest/minibatch.html>`_.
-  ``--dynet-gpus NUMBER``: Specify how many GPUs you want to use, if
   DyNet is compiled with CUDA.
-  ``--dynet-gpu``: Specify whether to use GPU or not. Note that it is an option for Python programs.
-  ``--dynet-devices CPU,GPU:1,GPU:3,GPU:0``: Specify the CPU/GPU devices that you
   want to use. You can the physical ID for GPU and can not specify the ID for CPU.
   This is an useful option working together with your multi-device code.
   Currently, DyNet needs you to specify the device ID explictly.
   The option ``--dynet-gpu-ids`` is deprecated.
-  ``--dynet-profiling NUMBER``: Will output information about the amount of
   time/memory used by each node in the graph. Profile level with ``0, 1`` and ``2``.
