Saving and Loading
~~~~~~~~~~~~~~~~~~

DyNet provides C++ interfaces for users to save and restore model parameters. The user has two options for saving a model. In the most basic use case, a complete ``ParameterCollection`` object can be saved. At loading time, the user should define and allocate the same parameter variables that were present in the model when it was saved (this usually amounts to having the same parameter creation called by both code paths), and then call ``populate`` and pass in the ``ParameterCollection`` object containing the parameters that should be loaded.

.. code:: cpp

    #include <dynet/io.h>
    // save end
    ParameterCollection m;
    Parameter a = m.add_parameters({100});
    LookupParameter b = m.add_lookup_parameters(10, {100});
    Parameter c = m.add_parameters({1000});
    {
        dynet::TextFileSaver s("/tmp/tmp.model");
        s.save(m);
    }

    // load end
    ParameterCollection m;
    m.add_parameters({100});
    m.add_lookup_parameters(10, {100});
    m.add_parameters({1000});
    {
        dynet::TextFileLoader l("/tmp/tmp.model");
        l.populate(m);
    }


However, in some cases it is useful to save only a subset of parameter objects(for example, if users wish to load these in a pretraining setup). Here, ``Parameter`` or ``LookupParameter`` objects can be saved explicitly. User could also specify keys for partial saving and loading.

.. code:: cpp

    #include <dynet/io.h>
    // save end
    ParameterCollection m1, m2;
    m1.add_parameters({10}, "a");
    m1.add_lookup_parameters(10, {2}, "la");
    Parameter param_b = m2.add_parameters({3, 7});
    {
        dynet::TextFileSaver s("/tmp/tmp.model");
        s.save(m1, "/namespace_tmp/");
        s.save(param_b, "param_b");
    }

    // load end
    ParameterCollection m;
    m.add_parameters({10});
    m.add_lookup_parameters(10, {2});
    {
        dynet::TextFileLoader l("/tmp/tmp.model");
        l.populate(m, "/namespace_tmp/");
        Parameter param_b = m.add_parameters({3, 7});
        l.populate(param_b, "param_b");
    }

    // load end
    // user can use equivalent interfaces to load model parameters
    ParameterCollection m;
    Parameter param_a, param_b;
    LookupParameter l_param;
    {
      dynet::TextFileLoader l("/tmp/tmp.model");
      param_a = l.load_param(m, "/namespace_tmp/a");
      l_param = l.load_lookup_param(m, "/namespace_tmp/la");
      param_b = l.load_param(m, "param_b");
    }


A word of warning: in previous versions of DyNet, Builder objects needed to be serialized. This is no longer the case. (The Python inerface does allow serialization of builder objects out of the box).

Currently, DyNet only supports plain text format. The native format is quite simple so very readable. The model file is consist of basic storage blocks. A basic block starts with a first line of meta data information: ``#object_type# object_name dimension block_size`` and the remaining part of real data. During loading process, DyNet uses meta data lines to locate the objects user wants to load.
