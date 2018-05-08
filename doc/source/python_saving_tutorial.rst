.. role:: python(code)
   :language: python

Saving Models
~~~~~~~~~~~~~
DyNet provides the ability to save and restore model parameters. The user has several options for saving and restoring parameters.

Saving an entire model
======================

The first option is to save the complete ``ParameterCollection`` object. At loading time, the user should define and allocate the same parameter objects that were present in the model when it was saved, and in the same order (this usually amounts to having the same parameter creation called by both code paths), and then call ``populate`` on the ``ParameterCollection`` object containing the parameters that should be loaded.

.. code:: python

    import dynet as dy
    # save
    m = dy.ParameterCollection()
    a = m.add_parameters(100)
    b = m.add_lookup_parameters((10, 100))
    c = m.add_parameters(1000)
    m.save("/tmp/tmp.model")

    # load 
    m2 = dy.ParameterCollection()
    x = m2.add_parameters(100);
    y = m2.add_lookup_parameters((10, 100))
    z = m2.add_parameters(1000)
    m2.populate("/tmp/tmp.model")


Partial Saving And Loading (Low-level API)
==========================================

 (This API follows the C++ partial saving and loading paradigm. See below for a higher level pythonic API.)

In some cases it is useful to save only a subset of parameter objects (for example, if users wish to load these in a pretraining setup). Here, ``Parameter`` or ``LookupParameter`` objects can be saved explicitly. User could also specify keys for partial saving and loading.


.. code:: python

    import dynet as dy
    # save
    m1 = dy.ParameterCollection()   # m1.name() == "/"
    m2 = dy.ParameterCollection()   # m2.name() == "/"
    m3 = m1.add_subcollection("m3") # m3.name() == "/m3/"
    a = m1.add_parameters(10, name="a") # a.name() == "/a"
    L = m1.add_lookup_parameters((10, 2), name="la") # L.name() == "/la"
    param_b = m2.add_parameters((3, 7))              # param_b.name() == "/_0"
    param_c = m3.add_parameters((3, 7), name="pc")   # param_c.name() == "/m3/pc"
    param_d = m3.add_parameters((3, 7))              # param_d.name() == "/m3/_0"
    L.save("/tmp/tmp.model", "/X") # ignores L.name(), saves L under "/X"
    a.save("/tmp/tmp.model", append=True) # uses a.name()
    param_c.save("/tmp/tmp.model", append=True)
    param_b.save("/tmp/tmp.model", append=True)
    param_d.save("/tmp/tmp.model", append=True)

    # load
    m = dy.ParameterCollection()
    a2 = m.add_parameters(10)
    L2 = m.add_lookup_parameters((10, 2))
    c = m.add_parameters((3,7))
    L2.populate("/tmp/tmp.model", "/X")
    a.populate("/tmp/tmp.model", "/a")
    c.populate("/tmp/tmp.model", "/m3/pc")


(See the documentation of ``ParameterCollection`` for further information about ``sub_collections`` and the use of collection hierarchies )

----

One can also save and load builder objects using their internal parameter collection.

.. code:: python

    # save
    lstm = dy.LSTMBuilder(2, 100, 100, m1) 
    pc = lstm.param_collection()  # pc.name() == "/lstm-builder/"

    lstm2 = dy.LSTMBuilder(2, 50, 50, m1) 
    pc2 = lstm2.param_collection()  # pc2.name() == "/lstm-builder_1/"

    pc2.save("/tmp/tmp.model",append=False)
    pc.save("/tmp/tmp.model",append=True)

    # load
    lstm2 = dy.LSTMBuilder(2, 50, 50, m) 
    lstm2.param_collection().populate("/tmp/tmp.model", "/lstm-builder_1/")

    lstm = dy.LSTMBuilder(2, 100, 100, m) 
    lstm.param_collection().populate("/tmp/tmp.model", "/lstm-builder/")



Partial Saving And Loading (High-level API)
===========================================

Use the module level ``dy.save(basename, lst)`` and ``dy.load(basename, param_collection)`` methods. 

``dy.save`` gets a base filename and a list of saveable objects (see below), and saves them to file.

``dy.load`` gets a base filename and a parameter collection (model), and returns a
list of objects, in the same order that were passed to ``dy.save``. The paramters
of the objects are added to the model.

Notice that you do not need to specify sizes when loading.

.. code:: python

    import dynet as dy

    pc = dy.ParameterCollection()
    W = pc.add_parameters((100,50))
    E = pc.add_lookup_parameters((1000,50))
    builder_a = dy.LSTMBuilder(2, 50, 50, pc)
    builder_b = dy.LSTMBuilder(2, 100, 100, pc)

    dy.save("/tmp/model", [E, builder_b, W])
    # this will create two files, "/tmp/model.data" and "/tmp/model.meta"

    # then, when loading:
    pc2 = dy.ParameterCollection()
    E2, builder2, W2 = dy.load("/tmp/model", pc2)

What can be saved?
------------------

Each object in ``lst`` must be one of the following:

1. Parameter
2. LookupParameter
3. One of the built-in types (VanillaLSTMBuilder, LSTMBuilder, GRUBuilder, SimpleRNNBuilder, BiRNNBuilder)
4. A type adhering to the following interface:

  - has a ``.param_collection()`` method returning a ParameterCollection object with the parameters in the object.
  - has a pickleable ``.spec`` property with items describing the object
  - has a ``.from_spec(spec, model)`` static method that will create and return a new instane of the object with the needed parameters/etc.

Note, the built-in types in (3) above can be saved/loaded this way simply because 
they support this interface.

behind the scenes:

- for each item, we write to ``basename.meta``:

 - if its a Parameters/ParameterCollection: 
      its type and full name.

 - if its a builder:
      its class, its spec, the full name of its parameters collection.

- the associated parameters/sub-collection is then saved to ``.data``

Example of a user-defined saveable type:
----------------------------------------

.. code:: python

  # Example of a user-defined saveable type.
  class OneLayerMLP(object):
    def __init__(self, model, num_input, num_hidden, num_out, act=dy.tanh):
      pc =  model.add_subcollection()
      self.W1 = pc.add_parameters((num_hidden, num_input))
      self.W2 = pc.add_parameters((num_out, num_hidden))
      self.b1 = pc.add_parameters((num_hidden))
      self.b2 = pc.add_parameters((num_out))
      self.pc = pc
      self.act = act
      self.spec = (num_input, num_hidden, num_out, act)

    def __call__(self, input_exp):
      W1 = dy.parameter(self.W1)
      W2 = dy.parameter(self.W2)
      b1 = dy.parameter(self.b1)
      b2 = dy.parameter(self.b2)
      g = self.act
      return dy.softmax(W2*g(W1*input_exp + b1)+b2)
      
    # support saving:
    def param_collection(self): return self.pc
      
    @staticmethod
    def from_spec(spec, model):
      num_input, num_hidden, num_out, act = spec
      return OneLayerMLP(model, num_input, num_hidden, num_out, act)

And for the usage:

.. code:: python

  import dynet as dy
  m = dy.ParameterCollection()
  # create an embedding table.
  E = m.add_lookup_parameters((1000,10))
  # create an MLP from 10 to 4 with a hidden layer of 20.
  mlp = OneLayerMLP(m, 10, 20, 4, dy.rectify)

  # use them together.
  output = mlp(E[3])

  # now save the model:
  dy.save("basename",[mlp, E])

  # now load:
  m2 = dy.ParameterCollection()
  mlp2, E2 = dy.load("basename", m2)

  output2 = mlp2(E2[3])

  import numpy
  assert(numpy.array_equal(output2.npvalue(), output.npvalue()))

File format
===========

Currently, DyNet only supports plain text format. The native format is quite simple so very readable. The model file is consist of basic storage blocks. A basic block starts with a first line of meta data information: ``#object_type# object_name dimension block_size`` and the remaining part of real data. During loading process, DyNet uses meta data lines to locate the objects user wants to load.

In the pythonic high-level partial saving/loading API, the ``.data`` file adheres to
the format above, while the ``.meta`` file conains information on objects types and sizes (for the specifics of the ``.meta`` file format see code of ``_save_one`` and ``_load_one`` in ``_dynet.pyx``).
