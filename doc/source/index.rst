.. Dynet documentation master file, created by
   sphinx-quickstart on Thu Oct 13 16:13:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dynet documentation
=================================

DyNet (formerly known as `cnn <http://github.com/clab/cnn-v1>`_) is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python) and is designed to be efficient when run on either CPU or GPU, and to work well with networks that have dynamic structures that change for every training instance. For example, these kinds of networks are particularly important in natural language processing tasks, and DyNet has been used to build state-of-the-art systems for `syntactic parsing <https://github.com/clab/lstm-parser>`_, `machine translation <https://github.com/neubig/lamtram>`_, `morphological inflection <https://github.com/mfaruqui/morph-trans>`_, and many other application areas.

DyNet can be installed according to the instructions below:

.. toctree::
   :maxdepth: 2

   install
   python

And get the basic information to create programs and use models:

.. toctree::
   :maxdepth: 3

   tutorial
   commandline
   operations
   builders
   optimizers
   examples

Mode advanced topics are below:

.. toctree::
   :maxdepth: 2

   minibatch
   multiprocessing
   unorthodox

And we welcome your contributions!

.. toctree::
   :maxdepth: 2

   contributing
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

