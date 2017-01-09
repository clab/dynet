.. DyNet documentation master file, created by
   sphinx-quickstart on Thu Oct 13 16:13:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DyNet documentation
=================================

`DyNet <http://github.com/clab/dynet>`_ (formerly known as `cnn <http://github.com/clab/cnn-v1>`_) is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python) and is designed to be efficient when run on either CPU or GPU, and to work well with networks that have dynamic structures that change for every training instance. For example, these kinds of networks are particularly important in natural language processing tasks, and DyNet has been used to build state-of-the-art systems for `syntactic parsing <https://github.com/clab/lstm-parser>`_, `machine translation <https://github.com/neubig/lamtram>`_, `morphological inflection <https://github.com/mfaruqui/morph-trans>`_, and many other application areas.

Read the documentation below to get started, and feel free to contact the `dynet-users <https://groups.google.com/forum/#!forum/dynet-users>`_ group with any questions (if you want to receive email make sure to select "all email" when you sign up). We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull request through the `github page <http://github.com/clab/dynet>`_.

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

