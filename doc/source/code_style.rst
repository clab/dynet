Coding Tips and Style
=====================

Coding Tips
-----------

**Adding New Operations:**
One of the most common things that one will want to do to modify DyNet is to add a new operation
to calculate a new function.
You can find more information on how to do so at the end of the tutorial slides
`here <http://phontron.com/slides/emnlp2016-dynet-tutorial-part1.pdf>`_.

Coding Conventions
------------------

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

**Throwing Exceptions:** When the user does something illegal, throw an
exception. "assert" should never be used for something that might be triggered
by a user. (As `noted <https://github.com/clab/dynet/issues/139>`_)

