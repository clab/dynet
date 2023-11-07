<div align="center">
  <img alt="DyNet" src="doc/source/images/dynet_logo.png"><br><br>
</div>

---

[![Build Status (Travis CI)](https://travis-ci.org/clab/dynet.svg?branch=master)](https://travis-ci.org/clab/dynet)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/clab/dynet?svg=true)](https://ci.appveyor.com/project/danielh/dynet-c3iuq)
[![Build Status (Docs)](https://readthedocs.org/projects/dynet/badge/?version=latest)](http://dynet.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/dyNET.svg)](https://badge.fury.io/py/dyNET)

The Dynamic Neural Network Toolkit

- [General](#general)
- [Installation](#installation)
  - [C++](#c-installation)
  - [Python](#python-installation)
- [Getting Started](#getting-started)
- [Citing](#citing)
- [Releases and Contributing](#releases-and-contributing)


## General

DyNet is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python) and is designed to be efficient when run on either CPU or GPU, and to work well with networks that have dynamic structures that change for every training instance. For example, these kinds of networks are particularly important in natural language processing tasks, and DyNet has been used to build state-of-the-art systems for [syntactic parsing](https://github.com/clab/lstm-parser), [machine translation](https://github.com/neubig/lamtram), [morphological inflection](https://github.com/mfaruqui/morph-trans), and many other application areas.

Read the [documentation](http://dynet.readthedocs.io/en/latest/) to get started, and feel free to contact the [dynet-users group](https://groups.google.com/forum/#!forum/dynet-users) group with any questions (if you want to receive email make sure to select "all email" when you sign up). We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull request through the [github page](http://github.com/clab/dynet).

You can also read more technical details in our [technical report](https://arxiv.org/abs/1701.03980).

## Getting started

You can find tutorials about using DyNet [here (C++)](http://dynet.readthedocs.io/en/latest/tutorial.html#c-tutorial) and [here (python)](http://dynet.readthedocs.io/en/latest/tutorial.html#python-tutorial), and [here (EMNLP 2016 tutorial)](https://github.com/clab/dynet_tutorial_examples).

One aspect that sets DyNet apart from other tookits is the **auto-batching** feature. See the [documentation](http://dynet.readthedocs.io/en/latest/minibatch.html) about batching.

The `example` folder contains a variety of examples in C++ and python.


## Installation

DyNet relies on a number of external programs/libraries including CMake and
Eigen. CMake can be installed from standard repositories.

For example on **Ubuntu Linux**:

    sudo apt-get install build-essential cmake

Or on **macOS**, first make sure the Apple Command Line Tools are installed, then
get CMake, and Mercurial with either homebrew or macports:

    xcode-select --install
    brew install cmake  # Using homebrew.
    sudo port install cmake # Using macports.

On **Windows**, see [documentation](http://dynet.readthedocs.io/en/latest/install.html#windows-support).

To compile DyNet you also need a [specific version of the Eigen
library](https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip). **If you use any of the
released versions, you may get assertion failures or compile errors.**
You can get it easily using the following command:

    mkdir eigen
    cd eigen
    wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
    unzip eigen-b2e267dc99d4.zip


### C++ installation

You can install dynet for C++ with the following commands

    # Clone the github repository
    git clone https://github.com/clab/dynet.git
    cd dynet
    mkdir build
    cd build
    # Run CMake
    # -DENABLE_BOOST=ON in combination with -DENABLE_CPP_EXAMPLES=ON also
    # compiles the multiprocessing c++ examples
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DENABLE_CPP_EXAMPLES=ON
    # Compile using 2 processes
    make -j 2
    # Test with an example
    ./examples/xor

For more details refer to the [documentation](http://dynet.readthedocs.io/en/latest/install.html#building)

### Python installation

You can install DyNet for python by using the following command

    pip install git+https://github.com/clab/dynet#egg=dynet

For more details refer to the [documentation](http://dynet.readthedocs.io/en/latest/python.html#installing-dynet-for-python)

## Citing

If you use DyNet for research, please cite this report as follows:

    @article{dynet,
      title={DyNet: The Dynamic Neural Network Toolkit},
      author={Graham Neubig and Chris Dyer and Yoav Goldberg and Austin Matthews and Waleed Ammar and Antonios Anastasopoulos and Miguel Ballesteros and David Chiang and Daniel Clothiaux and Trevor Cohn and Kevin Duh and Manaal Faruqui and Cynthia Gan and Dan Garrette and Yangfeng Ji and Lingpeng Kong and Adhiguna Kuncoro and Gaurav Kumar and Chaitanya Malaviya and Paul Michel and Yusuke Oda and Matthew Richardson and Naomi Saphra and Swabha Swayamdipta and Pengcheng Yin},
      journal={arXiv preprint arXiv:1701.03980},
      year={2017}
    }


## Contributing

We welcome any contribution to DyNet! You can find the contributing guidelines [here](http://dynet.readthedocs.io/en/latest/contributing.html)
