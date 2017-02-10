.. _minibatching:

Minibatching
============

Minibatching Overview
---------------------

Minibatching takes multiple training examples and groups them together to be processed simultaneously, often allowing for large gains in computational efficiency due to the fact that modern hardware (particularly GPUs, but also CPUs) have very efficient vector processing instructions that can be exploited with appropriately structured inputs.
As shown in the figure below, common examples of this in neural networks include grouping together matrix-vector multiplies from multiple examples into a single matrix-matrix multiply, or performing an element-wise operation (such as ``tanh``) over multiple vectors at the same time as opposed to processing single vectors individually.

.. image:: images/minibatch.png
  :align: center

In most neural network toolkits, mini-batching is largely left to the user, with a bit of help from the toolkit.
This is usually done by adding an additional dimension to the tensor that they are interested in processing, and ensuring that all operations consider this dimension when performing processing.
This adds some cognitive load, as the user must keep track of this extra batch dimension in all their calculations, and also ensure that they use the correct ordering of the batch dimensions to achieve maximum computational efficiency.
Users must also be careful when performing operations that combine batched and unbatched elements (such as batched hidden states of a neural network and unbatched parameter matrices or vectors), in which case they must concatenate vectors into batches, or "broadcast" the unbatched element, duplicating it along the batch dimension to ensure that there are no illegal dimension mismatches.

DyNet hides much of this complexity from the user through the use of specially designed batching operations which treat the number of mini-batch elements not as another standard dimension, but as a special dimension with particular semantics.
Broadcasting is done behind the scenes by each operation implemented in DyNet, and thus the user must only think about inputting multiple pieces of data for each batch, and calculating losses using multiple labels.

First, let's take a look at a non-minibatched example using the Python API.
In this example, we look up word embeddings ``word_1`` and ``word_2`` using lookup parameters ``E``.
We then perform an affine transform using weights ``W`` and bias ``b``, and perform a softmax.
Finally, we calculate the loss given the true label ``out_label``.

.. code-block:: python

  # in_words is a tuple (word_1, word_2)
  # out_label is an output label
  word_1 = E[in_words[0]]
  word_2 = E[in_words[1]]
  scores_sym = W*dy.concatenate([word_1, word_2])+b
  loss_sym = dy.pickneglogsoftmax(scores_sym, out_label)

Next, let's take a look at the mini-batched version:

.. code-block:: python

  # in_words is a list [(word_{1,1}, word_{1,2}), (word_{2,1}, word_{2,2}), ...]
  # out_labels is a list of output labels [label_1, label_2, ...]
  word_1_batch = dy.lookup_batch(E, [x[0] for x in in_words])
  word_2_batch = dy.lookup_batch(E, [x[1] for x in in_words])
  scores_sym = W*dy.concatenate([word_1_batch, word_2_batch])+b
  loss_sym = dy.sum_batches( dy.pickneglogsoftmax_batch(scores_sym, out_labels) )

We can see there are only 4 major changes: the word IDs need to be transformed into lists of IDs instead of a single ID, we need to call ``lookup_batch`` instead of the standard lookup, we need to call ``pickneglogsoftmax_batch`` instead of the unbatched version, and we need to call ``sum_batches`` at the end to sum the loss from all the batches.

Comparison of Standard and Minibatched Functions
------------------------------------------------

(TODO: This documentation is not yet finished. We need a comparison of standard and mini-batched functions in the C++ and Python APIs.)
