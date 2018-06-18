Installing DyNet for C++
========================

How to build DyNet and link it with your C++ programs.

Prerequisites
-------------

DyNet relies on a number of external programs/libraries including CMake,
Eigen, and Mercurial (to install Eigen). CMake, and Mercurial can
be installed from standard repositories. 

For example on **Ubuntu Linux**:

::

    sudo apt-get install build-essential cmake mercurial

Or on **macOS**, first make sure the Apple Command Line Tools are installed, then
get CMake, and Mercurial with either homebrew or macports:

::

    xcode-select --install
    brew install cmake hg  # Using homebrew.
    sudo port install cmake mercurial # Using macports.

On **Windows**, see :ref:`windows-cpp-install`.

To compile DyNet you also need the `development version of the Eigen
library <https://bitbucket.org/eigen/eigen>`__. **If you use any of the
released versions, you may get assertion failures or compile errors.**
If you don't have Eigen already, you can get it easily using the
following command:

::

    hg clone https://bitbucket.org/eigen/eigen/ -r b2e267d
    
The `-r NUM` specified a revision number that is known to work.  Adventurous
users can remove it and use the very latest version, at the risk of the code
breaking / not compiling. On macOS, you can install the latest development
of Eigen using Homebrew:

::

    brew install --HEAD eigen

Building
--------

To get and build DyNet, clone the repository

::

    git clone https://github.com/clab/dynet.git

then enter the directory and use `cmake <http://www.cmake.org/>`__
to generate the makefiles. When you run ``cmake``, you will need to specify
the path to Eigen, and will probably want to specify ``ENABLE_CPP_EXAMPLES``
to compile the C++ examples.

::

    cd dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DENABLE_CPP_EXAMPLES=ON


Then compile, where "2" can be replaced by the number of cores on your
machine

::

    make -j 2

To see that things have built properly, you can run

::

    ./examples/train_xor

which will train a multilayer perceptron to predict the xor function.

If any process here fails, please see :ref:`debugging-asking` for help.

Compiling/linking external programs
-----------------------------------

When you want to use DyNet in an external program, you will need to add
the ``dynet`` directory to the compile path:

::

    -I/path/to/dynet

and link with the DyNet library:

::

    -L/path/to/dynet/build/dynet -ldynet

GPU/cuDNN/MKL support
---------------------

GPU (CUDA) support
~~~~~~~~~~~~~~~~~~

DyNet supports running programs on GPUs with CUDA. If you have CUDA
installed, you can build DyNet with GPU support by adding
``-DBACKEND=cuda`` to your cmake options. The linking method is exactly
the same as with the CPU backend case.

::

    -L/path/to/dynet/build/dynet -ldynet

If you know the CUDA architecture supported by your GPU (e.g. by referencing
`this page <http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`__)
you can speed compilation significantly by adding ``-DCUDA_ARCH=XXX`` where
``XXX`` is your architecture number.

cuDNN support
~~~~~~~~~~~~~

When running DyNet with CUDA on GPUs, some of DyNet's functionality
(e.g. conv2d) depends on the `NVIDIA cuDNN libraries <https://developer.nvidia.com/cudnn>`__.
CMake will automatically detect cuDNN in the CUDA installation path 
(i.e. ``/usr/local/cuda``) and enable it if detected.

If CMake is unable to find cuDNN automatically, try setting `CUDNN_ROOT`, such as

::

    -DCUDNN_ROOT="/path/to/CUDNN"

However, if you don't have cuDNN installed, the dependent functionality
will be automatically disabled and an error will be throwed during runtime if you try
to use them.

MKL support
~~~~~~~~~~~

DyNet can leverage Intel's MKL library to speed up computation on the CPU.
As an example, we've seen 3x speedup in seq2seq training when using MKL. To use MKL, include the following cmake option:

::

    -DMKL=TRUE

If CMake is unable to find MKL automatically, try setting `MKL_ROOT`, such as

::

    -DMKL_ROOT="/path/to/MKL"

One common install location is ``/opt/intel/mkl/``.

If either `MKL` or `MKL_ROOT` are set, CMake will look for MKL.

By default, MKL will use all CPU cores. You can control how many cores MKL uses by setting the environment
variable `MKL_NUM_THREADS` to the desired number. The following is the total time to process 250 training 
examples running the example encdec (on a 6 core Intel Xeon E5-1650):

::

    encdec.exe --dynet-seed 1 --dynet-mem 1000 train-hsm.txt dev-hsm.txt
 
::

    +-----------------+------------+---------+
    | MKL_NUM_THREADS | Cores Used | Time(s) |
    +-----------------+------------+---------+
    | <Without MKL>   |     1      |  28.6   |
    |       1         |     1      |  13.3   |
    |       2         |     2      |   9.5   |
    |       3         |     3      |   8.1   |
    |       4         |     4      |   7.8   |
    |       6         |     6      |   8.2   |
    +-----------------+------------+---------+

As you can see, for this particular example, using MKL roughly doubles the speed of computation while 
still using only one core. Increasing the number of cores to 2 or 3 is quite beneficial, but beyond that
there are diminishing returns or even slowdown.

Compiling with Boost
~~~~~~~~~~~~~~~~~~~~

DyNet requires Boost for a few pieces of less-commonly-used functionality
to be enabled (unit tests and multi-processing). Boost can be enabled by using the
``-DENABLE_BOOST=ON`` flag to ``cmake``. In general, DyNet will find
Boost it if it is in the standard
location. If Boost is in a non-standard location, say ``$HOME/boost``,
you can specify the location by adding the following to your CMake
options:

::

    -DBOOST_ROOT:PATHNAME=$HOME/boost -DBoost_LIBRARY_DIRS:FILEPATH=$HOME/boost/lib
    -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE

Note that you will also have to set your ``LD_LIBRARY_PATH``(``DYLD_LIBRARY_PATH`` instead for osx) to point to
the ``boost/lib`` directory.
Note also that Boost must be compiled with the same compiler version as
you are using to compile DyNet.

.. _windows-cpp-install:

Windows Support
---------------

DyNet has been tested to build in Windows using Microsoft Visual Studio
2015. You may be able to build with MSVC 2013 by slightly modifying the
instructions below.

First, install Eigen following the above instructions.

To generate the MSVC solution and project files, run
`cmake <http://www.cmake.org>`__, pointing it to the location you
installed Eigen (for example, at c:\\libs\\Eigen):

::

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=c:/libs/Eigen -G"Visual Studio 14 2015 Win64"

This will generate `dynet.sln`. Simply open this and build all. **Note: multi-process functionality is
currently not supported in Windows, so the multi-process examples (`*-mp`) will not be included
in the generated solution**

The Windows build also supports MKL and CUDA with the latest version of Eigen. If you build with 
CUDA and/or cuDNN, ensure their respective DLLs are in your PATH environment variable when you use
dynet (whether in native C++ or Python). For example:

::

    set PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin";"c:\libs\cudnn-8.0-windows10-x64-v5.1\bin";%PATH%


