Python Reference Manual
=======================


DyNet global parameters
-----------------------

DynetParams
~~~~~~~~~~~

.. autoclass:: dynet.DynetParams
   :members:
   :show-inheritance:

Initialization functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dynet.init

.. autofunction:: dynet.init_from_params

.. autofunction:: dynet.reset_random_seed

ParameterCollection and Parameters
----------------------------------

ParameterCollection
~~~~~~~~~~~~~~~~~~~
.. autoclass:: dynet.ParameterCollection
   :members:
   :show-inheritance:

Parameters and LookupParameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dynet.Parameters
   :members:
   :show-inheritance:

.. autoclass:: dynet.LookupParameters
   :members:
   :show-inheritance:

Parameters initializers
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dynet.PyInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.NormalInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.UniformInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.ConstInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.IdentityInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.GlorotInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.SaxeInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.FromFileInitializer
   :members:
   :show-inheritance:

.. autoclass:: dynet.NumpyInitializer
   :members:
   :show-inheritance:

High level saving/loading
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dynet.save

.. autofunction:: dynet.load

Computation Graph
-----------------


.. autofunction:: dynet.renew_cg

.. autofunction:: dynet.cg_version

.. autofunction:: dynet.print_text_graphviz

.. autofunction:: dynet.cg_checkpoint

.. autofunction:: dynet.cg_revert

.. autofunction:: dynet.cg


.. autoclass:: dynet.ComputationGraph
   :members:
   :show-inheritance:

Operations
----------

Expressions
~~~~~~~~~~~

.. autoclass:: dynet.Expression
   :members:
   :show-inheritance:

Operations
~~~~~~~~~~

Operations are used to build expressions

Input operations
^^^^^^^^^^^^^^^^

.. autofunction:: dynet.inputTensor

.. autofunction:: dynet.sparse_inputTensor

.. autofunction:: dynet.parameter

.. autofunction:: dynet.const_parameter

.. autofunction:: dynet.scalarInput

.. autofunction:: dynet.vecInput

.. autofunction:: dynet.inputVector

.. autofunction:: dynet.matInput

.. autofunction:: dynet.inputMatrix

.. autofunction:: dynet.lookup

.. autofunction:: dynet.lookup_batch

.. autofunction:: dynet.zeros

.. autofunction:: dynet.ones

.. autofunction:: dynet.constant

.. autofunction:: dynet.random_normal

.. autofunction:: dynet.random_bernoulli

.. autofunction:: dynet.random_uniform

.. autofunction:: dynet.random_gumbel

.. autofunction:: dynet.noise

Arithmetic operations
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.cdiv

.. autofunction:: dynet.cmult

.. autofunction:: dynet.colwise_add

.. autofunction:: dynet.squared_norm

.. autofunction:: dynet.l2_norm

.. autofunction:: dynet.exp

.. autofunction:: dynet.square

.. autofunction:: dynet.sqrt

.. autofunction:: dynet.abs

.. autofunction:: dynet.erf

.. autofunction:: dynet.cube

.. autofunction:: dynet.log

.. autofunction:: dynet.log_sigmoid

.. autofunction:: dynet.lgamma

.. autofunction:: dynet.sin

.. autofunction:: dynet.cos

.. autofunction:: dynet.tan

.. autofunction:: dynet.asin

.. autofunction:: dynet.acos

.. autofunction:: dynet.atan

.. autofunction:: dynet.sinh

.. autofunction:: dynet.cosh

.. autofunction:: dynet.tanh

.. autofunction:: dynet.asinh

.. autofunction:: dynet.acosh

.. autofunction:: dynet.atanh

.. autofunction:: dynet.logistic

.. autofunction:: dynet.rectify

.. autofunction:: dynet.elu

.. autofunction:: dynet.selu

.. autofunction:: dynet.sparsemax

.. autofunction:: dynet.softsign

.. autofunction:: dynet.pow

.. autofunction:: dynet.bmin

.. autofunction:: dynet.bmax

.. autofunction:: dynet.cumsum

Reduction/moment operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.sum_elems

.. autofunction:: dynet.moment_elems

.. autofunction:: dynet.mean_elems

.. autofunction:: dynet.std_elems

.. autofunction:: dynet.sum_dim

.. autofunction:: dynet.moment_dim

.. autofunction:: dynet.mean_dim

.. autofunction:: dynet.std_dim

.. autofunction:: dynet.max_dim

.. autofunction:: dynet.min_dim

.. autofunction:: dynet.sum_batches

.. autofunction:: dynet.moment_batches

.. autofunction:: dynet.mean_batches

.. autofunction:: dynet.std_batches

.. autofunction:: dynet.fold_rows

.. autofunction:: dynet.esum

.. autofunction:: dynet.emax

.. autofunction:: dynet.logsumexp

.. autofunction:: dynet.average

Loss/Probability operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.softmax

.. autofunction:: dynet.log_softmax

.. autofunction:: dynet.pairwise_rank_loss

.. autofunction:: dynet.poisson_loss

.. autofunction:: dynet.huber_distance

.. autofunction:: dynet.pickneglogsoftmax

.. autofunction:: dynet.pickneglogsoftmax_batch

.. autofunction:: dynet.hinge

.. autofunction:: dynet.hinge_batch

.. autofunction:: dynet.kmh_ngram

.. autofunction:: dynet.squared_distance

.. autofunction:: dynet.l1_distance

.. autofunction:: dynet.binary_log_loss

Flow/Shaping operations
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.pick

.. autofunction:: dynet.pick_batch

.. autofunction:: dynet.pickrange

.. autofunction:: dynet.pick_batch_elem

.. autofunction:: dynet.pick_batch_elems

.. autofunction:: dynet.reshape

.. autofunction:: dynet.select_rows

.. autofunction:: dynet.select_cols

.. autofunction:: dynet.concatenate_cols

.. autofunction:: dynet.concatenate

.. autofunction:: dynet.concatenate_to_batch

.. autofunction:: dynet.nobackprop

.. autofunction:: dynet.flip_gradient

.. autofunction:: dynet.argmax

Noise operations
^^^^^^^^^^^^^^^^

.. autofunction:: dynet.noise

.. autofunction:: dynet.dropout

.. autofunction:: dynet.dropout_dim

.. autofunction:: dynet.dropout_batch

.. autofunction:: dynet.block_dropout

Linear algebra operations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.affine_transform

.. autofunction:: dynet.dot_product

.. autofunction:: dynet.transpose

.. autofunction:: dynet.inverse

.. autofunction:: dynet.trace_of_product

.. autofunction:: dynet.logdet

Convolution/Pooling operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.conv2d

.. autofunction:: dynet.conv2d_bias

.. autofunction:: dynet.maxpooling2d

.. autofunction:: dynet.filter1d_narrow

.. autofunction:: dynet.kmax_pooling

.. autofunction:: dynet.circ_conv

.. autofunction:: dynet.circ_corr

Tensor operations
^^^^^^^^^^^^^^^^^

**Remark**: Compiling the contraction operations takes a lot of time with CUDA. For this reason, only the CPU implementation is compiled by default. If you need those operations, you need to un-comment `this line <https://github.com/clab/dynet/blob/master/dynet/nodes-contract.cc#L11>`_ in the source before compiling. TODO: make this simpler.

.. autofunction:: dynet.contract3d_1d

.. autofunction:: dynet.contract3d_1d_bias

.. autofunction:: dynet.contract3d_1d_1d

.. autofunction:: dynet.contract3d_1d_1d_bias

Normalization operations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: dynet.layer_norm

.. autofunction:: dynet.weight_norm

Recurrent Neural Networks
-------------------------

RNN Builders
~~~~~~~~~~~~

.. autoclass:: dynet._RNNBuilder
   :members:
   :show-inheritance:
   :undoc-members:

.. autoclass:: dynet.SimpleRNNBuilder
   :members:
   :show-inheritance:

.. autoclass:: dynet.GRUBuilder
   :members:
   :show-inheritance:


.. autoclass:: dynet.VanillaLSTMBuilder
   :members:
   :show-inheritance:

.. autoclass:: dynet.CompactVanillaLSTMBuilder
   :members:
   :show-inheritance:

.. autoclass:: dynet.CoupledLSTMBuilder
   :members:
   :show-inheritance:

.. autoclass:: dynet.FastLSTMBuilder
   :members:
   :show-inheritance:

.. autoclass:: dynet.BiRNNBuilder
   :members:
   :show-inheritance:

RNN state
~~~~~~~~~

.. autoclass:: dynet.RNNState
   :members:
   :show-inheritance:

.. autoclass:: dynet.StackedRNNState
   :members:
   :show-inheritance:

Softmax Builders
----------------


.. autoclass:: dynet.SoftmaxBuilder
   :members:
   :show-inheritance:


.. autoclass:: dynet.StandardSoftmaxBuilder
   :members:
   :show-inheritance:


.. autoclass:: dynet.ClassFactoredSoftmaxBuilder
   :members:
   :show-inheritance:

Optimizers
----------

.. autoclass:: dynet.Trainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.SimpleSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.CyclicalSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.MomentumSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.AdagradTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.AdadeltaTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.RMSPropTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.AdamTrainer
   :members:
   :show-inheritance:

.. autoclass:: dynet.AmsgradTrainer
   :members:
   :show-inheritance:

MultiDevice
-----------

.. autofunction:: dynet.to_device

