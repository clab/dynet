.. DyNet documentation master file, created by
   sphinx-quickstart on Thu Oct 13 16:13:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DyNet documentation
===================

`DyNet <http://github.com/clab/dynet>`_ (formerly known as `cnn <http://github.com/clab/cnn-v1>`_) is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python) and is designed to be efficient when run on either CPU or GPU, and to work well with networks that have dynamic structures that change for every training instance. For example, these kinds of networks are particularly important in natural language processing tasks, and DyNet has been used to build state-of-the-art systems for `syntactic parsing <https://github.com/clab/lstm-parser>`_, `machine translation <https://github.com/neulab/xnmt>`_, `morphological inflection <https://github.com/mfaruqui/morph-trans>`_, and many other application areas.

Read the documentation below to get started. If you have any problems see :ref:`debugging` for how to debug and/or get in contact with the developers. We also greatly welcome contributions, so see :ref:`contributing` for details.

You can also read more technical details in our `technical report <https://arxiv.org/abs/1701.03980>`_. If you use DyNet for research, please cite this report as follows::

  @article{dynet,
    title={DyNet: The Dynamic Neural Network Toolkit},
    author={Graham Neubig and Chris Dyer and Yoav Goldberg and Austin Matthews and Waleed Ammar and Antonios Anastasopoulos and Miguel Ballesteros and David Chiang and Daniel Clothiaux and Trevor Cohn and Kevin Duh and Manaal Faruqui and Cynthia Gan and Dan Garrette and Yangfeng Ji and Lingpeng Kong and Adhiguna Kuncoro and Gaurav Kumar and Chaitanya Malaviya and Paul Michel and Yusuke Oda and Matthew Richardson and Naomi Saphra and Swabha Swayamdipta and Pengcheng Yin},
    journal={arXiv preprint arXiv:1701.03980},
    year={2017}
  }

DyNet can be installed according to the instructions below:

.. toctree::
   :maxdepth: 2

   install
   python
   other_languages

And get the basic information to create programs and use models:

.. toctree::
   :maxdepth: 3

   tutorial
   commandline
   debugging
   python_ref
   cpp_ref
   examples

Mode advanced topics are below:

.. toctree::
   :maxdepth: 2

   minibatch
   multiprocessing
   unorthodox
   projects

And we welcome your contributions!

.. toctree::
   :maxdepth: 2

   contributing
   citing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
