Projects using DyNet
==================

DyNet works for your complex neural networks
--------------------------------------------

DyNet was designed from the ground up to be fast for neural networks with complex structure or control flow such as the ones that you need to handle tree or graph structures, or perform reinforcement learning or training with exploration.
Below are some examples of full systems that use DyNet to handle their dynamic neural network needs.

Syntactic Parsing
-----------------

Parsing is currently the most prominent scenario in which DyNet has been used, and DyNet was behind the development of a number of methods such as `stack LSTMs <https://github.com/clab/lstm-parser>`_, `bi-directional LSTM feature extractors for dependency parsing <https://github.com/elikip/bist-parser>`_, `recurrent neural network grammars <https://github.com/clab/rnng>`_, and `hierarchical tree LSTMs <https://github.com/elikip/htparser>`_.
A `submission to the CoNLL shared task on dependency parsing <https://github.com/CoNLL-UD-2017/C2L2>`_ using DyNet registered second place, and was nearly an order of magnitude faster than other submissions.

Machine Translation
-------------------

DyNet is the backend chosen by a number of machine translation systems such as `Mantis <https://github.com/trevorcohn/mantis>`_, `Lamtram <https://github.com/neubig/lamtram>`_, `nmtkit <https://github.com/odashi/nmtkit>`_, and `xnmt <https://github.com/neulab/xnmt/>`_.
It has powered the development of models that use complicated structures, such as `lattice-to-sequence models <https://arxiv.org/abs/1704.00559>`_.

Speech Recognition
------------------

DyNet powers the "Listen, Attend, and Spell" style models in `xnmt <https://github.com/neulab/xnmt/>`_.
It has also been used to implement acoustic models using `connectionist temporal classification (CTC). <https://arxiv.org/pdf/1708.04469.pdf>`_

Graph Parsing
-------------

DyNet powers the `transition based UCCA parser <https://github.com/danielhers/tupa>`_ that can predict graph structures from text.

Language Modeling
-----------------

DyNet has been used in the development of `hybrid neural/n-gram language models <https://github.com/neubig/modlm>`_, and `generative syntactic language models <https://github.com/clab/rnng>`_.

Tagging
-------

DyNet supports applications to tagging for `named entity recognition <https://github.com/clab/stack-lstm-ner>`_, `semantic role labeling <https://github.com/clab/joint-lstm-parser>`_, `punctuation prediction <https://github.com/miguelballesteros/LSTM-punctuation>`_, and has been used in the creation of new architectures such as `segmental recurrent neural networks <https://github.com/clab/dynet/tree/master/examples/cpp/segrnn-sup>`_.

Morphology
----------

DyNet has been used in seminal work for `morphological inflection generation <https://github.com/mfaruqui/morph-trans>`_ and `inflection generation with hard attention <https://github.com/roeeaharoni/morphological-reinflection>`_.
