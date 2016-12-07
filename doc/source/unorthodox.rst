Unorthodox Design
=================

There are a couple design decisions about DyNet that are different from the way
things are implemented in other libraries, or different from the way you might
expect things to be implemented. The items below are a list of these unorthodox
design decisions, which you should read to avoid being surprised. We also try
to give some justification for these decisions (although we realize that this
is not the only right way to do things).

Sparse Updates
--------------

By default, DyNet parameter optimizers perform sparse updates over
``LookupParameters``. This means that if you have a ``LookupParameters``
object, use a certain subset of indices, then perform a parameter update, the
optimizer will loop over the used subset, and not perform any updates over
the unused values. This can improve efficiency in some cases: e.g. if you have
embeddings for a vocabulary of 100,000 words and you only use 5 of them in a
particular update, this will avoid doing updates over all 100,000. However,
there are two things to be careful of. First, this means that some update rules
such as ones using momentum such as ``MomentumSGDTrainer`` and ``AdamTrainer``
are not strictly correct (these could be made correct with some effort, but
this would complicate the programming interface, which we have opted against).
Also, on GPUs, because large operations are
relatively cheap, it can sometimes be faster to just perform a single operation
over all of the parameters, as opposed to multiple small operations. In this
case, you can set the ``sparse_updates_enabled`` variable of your ``Trainer``
to ``false``, and DyNet will perform a standard dense update, which is
guaranteed to be exactly correct, and potentially faster on GPU.

Weight Decay
------------

As described in the :ref:`command-line-options`, weight decay is implemented
through the option ``--dynet-weight-decay``. If this value is set to ``wd``,
each parameter in the model is multiplied by ``(1-wd)`` after every parameter
update. This weight decay is similar to L2 regularization, and is equivalent in
the case of using simple SGD (``SimpleSGDTrainer``), but it is *not the same*
when using any other optimizers such as ``AdagradTrainer`` or ``AdamTrainer``.
You can still try to use weight decay with these optimizers, and it might work,
but if you really want to correctly apply L2 regularization with these
optimizers, you will have to directly calculate the L2 norm of each of the
parameters and add it to the objective function before performing your update.

Minibatching Implementation
---------------------------

:ref:`minibatching` in DyNet is different than how it is implemented in other
libraries. In other libraries, you can create minibatches by explicitly adding
another dimension to each of the variables that you want to process, and
managing them yourself. Instead, DyNet provides special :ref:`operations` that
allow you to perform input, lookup, or loss calculation over mini-batched
input, then DyNet will handle the rest. The programming paradigm is a bit
different from other toolkits, and may take a bit of getting used to, but is
often more convenient once you're used to it.

Dropout Scaling
---------------

When using dropout to help prevent overfitting, dropout is generally applied
at training time, then at test time all the nodes in the neural net are used
to make the final decision, increasing robustness. However, because there is
a disconnect between the number of nodes being used in each situation, it is
important to scale the values of the output to ensure that they match in both
situations. There are two ways to do this:

* **Vanilla Dropout:** At training time, perform dropout with probability
  ``p``. At test time, scale the outputs of each node by ``p``.
* **Inverted Dropout:** At training time, perform dropout with probability
  ``p``, *and* scale the outputs by ``1/p``. At test time, use the outputs
  as-is.

The first is perhaps more common, but the second is convenient, because we
only need to think about dropout at training time, and thus DyNet opts to
use the latter. See `here <http://cs231n.github.io/neural-networks-2/#reg>`_
for more details on these two methods.
