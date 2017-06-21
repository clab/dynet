Saving and Loading
~~~~~~~~~~~~~~~~~~

Dynet provides straightforward C++ interfaces for users to save and restore the model parameters. In the saving side, user could either save a whole `ParameterCollection` object, or save some selected `parameters` or `lookup_parameters` in a `ParameterCollection` they want to save. In the loading side, user should define and allocate the variables in advance if they'd like to use `populate` interface. Dynet also supports partial load.

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


To be noticed, we don't support saving/loading a builder object any more. In other words, users should be responsible for saving/loading model parameters themselves.

Currently, dynet only supports the plain text io format upon files. The native format is quite simple so very readable. The model file is consist of basic storage blocks. A basic block starts with a first line of meta data information: `#object_type# object_name dimension block_size` and the remaining part of real data. During loading process, Dynet uses meta data lines to locate the objects user wants to load.
