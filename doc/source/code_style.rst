Coding Tips and Style
=====================

Coding Practices
----------------

**Testing:**
Before committing any code, tests should be run to make sure that the new code didn't break anything.
This can be done by using the ``make test`` command.
It is also highly recommended that you add unit tests for any new functionality.
Unit tests are implemented in the ``tests`` directory.
When making a bug fix, you can add a test that broke before the fix but passes afterwards.

That being said, tests are not an absolute requirement, so if you have a contribution but aren't sure
how to do tests, please don't let this stop you from contributing.

Coding Style Conventions
------------------------

DyNet (the main version in C++) has certain coding style standards:

**Overall Philosophy:** DyNet is designed to minimize the computational
overhead when creating networks. Try to avoid doing slow things like creating
objects or copying memory in places that will be called frequently during
computation graph construction.

**Function Names:** Function names are written in "snake_case".

**const:** Always use const if the input to a function is constant.

**Pointer vs. Reference:** When writing functions, use the following guidelines
(quoted from `here <http://stackoverflow.com/questions/114180/pointer-vs-reference/114351#114351>`_):

* Only pass a value by pointer if the value 0/NULL is a valid input in the
  current context.
* If a function argument is an out-value, then pass it by reference.
* Choose "pass by value" over "pass by const reference" only if the value is a
  POD (`Plain Old Datastructure <http://stackoverflow.com/questions/146452/what-are-pod-types-in-c>`_)
  or small enough (memory-wise) or in other ways cheap enough (time-wise) to
  copy.

**Error handling:** The C++ core of DyNet provides a mechanism for error handling that
should be used in all code. It consists of 3 macros as follows (included in ``globals.h``):

* ``DYNET_INVALID_ARG(msg)``: This is used to throw an error that is triggered when
  a user passes an invalid argument to one of the functions.
* ``DYNET_RUNTIME_ERR(msg)``: This is used to throw an error that could be triggered
  by a user, but is not the result of an invalid argument. For example, it could be
  used when something is not implemented yet, or when the program dies due to lack
  of memory, etc.
* ``DYNET_ASSERT(expr,msg)``: This is to be used to check things that should only
  happen due to a programming error within DyNet itself, and should never be
  triggered by a user. ``expr`` is a condition, and ``msg`` is a message explaining
  the exception, with ``ostream``-style formatting.

Coding Tips/How To
------------------

Adding New Operations
~~~~~~~~~~~~~~~~~~~~~

One of the most common things that one will want to do to modify DyNet is to add a new operation
to calculate a new function.
You can find more information on how to do so at the end of the tutorial slides
`here <http://phontron.com/slides/emnlp2016-dynet-tutorial-part1.pdf>`_ (note that some file
names are old).

Taking a look at the existing operations in the ``nodes-XXX.h`` and ``nodes-XXX.cc`` files
will be the best guide in creating new operations. Here are some fine-grained tips for
those that want to dive into the process.

1. ``fx`` is a pointer to the (preallocated) location for the result
   of forward to be stored
2. ``fx`` is not initialized, so after calling forward ``fx`` must contain the correct answer
3. dEdxi MUST **ACCUMULATE** a result since multiple calls to forward may depend on
   the same ``x_i``. Even, e.g., Identity must be implemented as ``dEdx1 += dEdf``.
4. scalars results of forward are placed in ``fx.v[0]``
5. DyNet manages its own memory, not Eigen, and it is configured with the
   EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate
   memory, it is (probably) because of an implicit creation of a temporary variable.
   If you really do need a temporary variable, its capacity must be requested by
   Node::aux_storage_size

And here are some notes on debugging problems with new operations

1. fx is uninitialized when forward is called- are you relying on it being 0?
2. dEdxi must accumulate (see point 3 above!)

Decreasing Compile Time
~~~~~~~~~~~~~~~~~~~~~~~

DyNet has a ``GPU_NUMFILES`` option that allows you to specify the number of separate
files that are compiled when compiling for GPU. This is set to 4 on Linux/Mac and 1 on
Windows (because MSVC doesn't support parallel compilation for GPU code). If you're
developing new operations for DyNet, it might be a good idea to set ``GPU_NUMFILES``
to zero, which will result in all ``nodes-XXX.cc`` files being compiled separately.
In this case, if you change a single file, it will only recompile that file instead
of recompiling all of the code in all of the ``nodes-XXX.cc`` files.
