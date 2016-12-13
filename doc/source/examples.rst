Examples
========

Here are some simple models coded in the examples of Dynet. Feel free to use and modify them.

Feed-forward models
-------------------

Although Dynet was primarily built for natural language processing purposes it is still possible to code feed-forward nets. Here are some bricks and examples to do so.

.. doxygengroup:: ffbuilders
    :members:
    :content-only:

Language models
---------------

Language modelling is one of the cornerstones of natural language processing. Dynet allows great flexibility in the creation of neural language models. Here are some examples.

.. doxygengroup:: lmbuilders
    :content-only:
    :members:

Sequence to sequence models
---------------------------

Dynet is well suited for the variety of sequence to sequence models used in modern NLP. Here are some pre-coded structs implementing the most common one.

.. doxygengroup:: seq2seqbuilders
    :members:
    :content-only:

