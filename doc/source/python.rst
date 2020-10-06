Installing DyNet for Python
===========================

Installing a Release CPU Version
--------------------------------

Python bindings to DyNet are supported for both Python 2.x and 3.x.
If you want to install a release version of DyNet and don't need to run on GPU, you can
simply run

::

    pip install dynet


Installing a Cutting-edge and/or GPU Version
--------------------------------------------

If you want the most recent features of DyNet from the development branch, or want GPU
compute capability, you'll want to install DyNet from source.

Before doing so, you will need to make sure that several packages are installed.
For example on **Ubuntu Linux**:

::
    
    sudo apt-get update
    sudo apt-get install python-pip build-essential cmake mercurial

Or on **macOS**, first make sure the Apple Command Line Tools are installed, then
get CMake, and Mercurial with either homebrew or macports:

::

    xcode-select --install
    brew install cmake hg python # Using homebrew.
    sudo port install cmake mercurial py-pip # Using macports.

On **Windows**, see :ref:`windows-python-install`.

(Currently, since the pip installation will build from source, you need to install
 cython ahead: ``pip install cython``.)

Once these packages are installed, the following will download, build and install
DyNet. Note that compiling DyNet may take a long time, up to 10 minutes or more, but as
long as you see "Running setup.py install for dynet" with the moving progress
wheel, things should be running.

.. code:: bash

    pip install git+https://github.com/clab/dynet#egg=dynet

If you have CUDA installed on your system and want to install with GPU support, you
can instead run the following command.

.. code:: bash

    BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet

Alternatively, you can add the following to your `requirements.txt` (for CUDA support
you will need to make sure that `BACKEND=cuda` is in your environmental variables when
DyNet is installed):

.. code:: bash

    git+https://github.com/clab/dynet#egg=dynet

You can also manually set the directory of the cuDNN library as follows:

.. code:: bash

    CUDNN_ROOT=/path/to/cudnn BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet

If installation using `pip` fails, if you copy-and-paste the entire log that you
get after running the `pip` command into a `github issue <https://github.com/clab/dynet/issues>`_,
we will help you debug. You can also try to install DyNet manually as listed below.

Manual Installation
-------------------

The following is a list of all the commands needed to perform a manual install:

.. code:: bash

    # Installing Python DyNet:

    pip install cython  # if you don't have it already.
    mkdir dynet-base
    cd dynet-base
    # getting dynet and eigen
    git clone https://github.com/clab/dynet.git
    mkdir eigen
    cd eigen
    wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
    unzip eigen-b2e267dc99d4.zip
    cd ../dynet
    mkdir build
    cd build
    # without GPU support (if you get an error that Eigen cannot be found, try using the full path to Eigen)
    cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python`
    # or with GPU support (if you get an error that Eigen cannot be found, try using the full path to Eigen)
    cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python` -DBACKEND=cuda

    make -j 2 # replace 2 with the number of available cores
    cd python
    python ../../setup.py build --build-dir=.. --skip-build install # add `--user` for a user-local install.
    
    # this should suffice, but on some systems you may need to add the following line to your
    # init files in order for the compiled .so files be accessible to Python.
    # /path/to/dynet/build/dynet is the location in which libdynet.dylib resides.
    export DYLD_LIBRARY_PATH=/path/to/dynet/build/dynet/:$DYLD_LIBRARY_PATH
    # if the environment is Linux, use LD_LIBRARY_PATH instead.
    export LD_LIBRARY_PATH=/path/to/dynet/build/dynet/:$LD_LIBRARY_PATH


To explain these one-by-one, first we get DyNet:

.. code:: bash

    cd $HOME
    mkdir dynet-base
    cd dynet-base
    git clone https://github.com/clab/dynet.git
    cd dynet
    git submodule init # To be consistent with DyNet's installation instructions.
    git submodule update # To be consistent with DyNet's installation instructions.

Then get Eigen:

.. code:: bash

    cd $HOME
    cd dynet-base
    mkdir eigen
    cd eigen
    wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
    unzip eigen-b2e267dc99d4.zip
    
We also need to make sure the ``cython`` module is installed. (you can
replace ``pip`` with your favorite package manager, such as ``conda``,
or install within a virtual environment)

.. code:: bash

    pip install cython

To simplify the following steps, we can set a bash variable to hold
where we have saved the main directories of DyNet and Eigen. In case you
have gotten DyNet and Eigen differently from the instructions above and
saved them in different location(s), these variables will be helpful:

.. code:: bash

    PATH_TO_DYNET=$HOME/dynet-base/dynet/
    PATH_TO_EIGEN=$HOME/dynet-base/eigen/

Compile DyNet.

This is pretty much the same process as compiling DyNet, with the
addition of the ``-DPYTHON=`` flag, pointing to the location of your
Python interpreter.

Assuming that the ``cmake`` command found all the needed libraries and
didn't fail, the ``make`` command will take a while, and compile DyNet
as well as the Python bindings. You can change ``make -j 2`` to a higher
number, depending on the available cores you want to use while
compiling.

You now have a working Python binding inside of ``build/dynet``. To
verify this is working:

.. code:: bash

    cd $PATH_TO_DYNET/build/python
    python

then, within Python:

.. code:: bash

    import dynet as dy
    print dy.__version__
    pc = dy.ParameterCollection()

In order to install the module so that it is accessible from everywhere
in the system, run the following:

.. code:: bash

    cd $PATH_TO_DYNET/build/python
    python ../../setup.py EIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN build --build-dir=.. --skip-build install --user

The ``--user`` switch will install the module in your local
site-packages, and works without root privileges. To install the module
to the system site-packages (for all users), or to the current `virtualenv`
(if you are on one), run ``python ../../setup.py EIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN build --build-dir=.. --skip-build install`` without this switch.

You should now have a working python binding (the ``dynet`` module).

Note however that the installation relies on the compiled DyNet library
being in ``$PATH_TO_DYNET/build/dynet``, so make sure not to move it
from there.

Now, check that everything works:

.. code:: bash

    cd $PATH_TO_DYNET
    cd examples/python
    python xor.py
    python rnnlm.py rnnlm.py

Alternatively, if the following script works for you, then your
installation is likely to be working:

::

    import dynet as dy
    pc = dy.ParameterCollection()

If it doesn't work and you get an error similar to the following:
::

    ImportError: dlopen(/Users/sneharajana/.python-eggs/dyNET-0.0.0-py2.7-macosx-10.11-intel.egg-tmp/_dynet.so, 2): Library not loaded: @rpath/libdynet.dylib
    Referenced from: /Users/sneharajana/.python-eggs/dyNET-0.0.0-py2.7-macosx-10.11-intel.egg-tmp/_dynet.so
    Reason: image not found``

then you may need to run the following (and add it to your shell init files):

.. code:: bash

    # OSX 
    export DYLD_LIBRARY_PATH=/path/to/dynet/build/dynet/:$DYLD_LIBRARY_PATH
    # Linux
    export LD_LIBRARY_PATH=/path/to/dynet/build/dynet/:$LD_LIBRARY_PATH

# /path/to/dynet/build/dynet is the location in which libdynet.so(libdynet.dylib under osx) resides.

Anaconda Support
----------------

`Anaconda 
<https://www.continuum.io/downloads>`_ is a popular package management system for Python, and DyNet can be installed into this environment.
First, make sure that you install all the necessary packages according to the instructions at the top of this page.
Then create an Anaconda environment and activate it as below:

::

     source activate my_environment_name

After this, you should be able to install using pip or manual installation as normal.

.. _windows-python-install:

Windows Support
---------------

You can also use Python on Windows, including GPU and MKL support. For simplicity, we recommend 
using a Python distribution that already has Cython installed. The following has been tested to work:

1) Install WinPython 2.7.10 (comes with Cython already installed).
2) Compile DyNet according to the directions in the Windows C++ documentation (:ref:`windows-cpp-install`), and additionally add the following flag when executing ``cmake``: ``-DPYTHON=/path/to/your/python.exe``.
3) Open a command prompt and set ``VS90COMNTOOLS`` to the path to your Visual Studio "Common7/Tools" directory. One easy way to do this is a command such as:

::

    set VS90COMNTOOLS=%VS140COMNTOOLS%

4) Open dynet.sln from this command prompt and build the "Release" version of the solution.
5) Follow the rest of the instructions above for testing the build and installing it for other users

Note, currently only the Release version works. Also, if you compile with CUDA and/or cuDNN, ensure
their respective DLLs are in your PATH environment variable when you run Python.

GPU/MKL Support
---------------

Installing on GPU
~~~~~~~~~~~~~~~~~

For installing on a computer with GPU, first install CUDA. The following
instructions assume CUDA is installed.

The installation process is pretty much the same, while adding the
``-DBACKEND=cuda`` flag to the ``cmake`` stage:

.. code:: bash

    cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DPYTHON=$PATH_TO_PYTHON -DBACKEND=cuda


If you know the CUDA architecture supported by your GPU (e.g. by referencing
`this page <http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`__)
you can speed compilation significantly by adding ``-DCUDA_ARCH=XXX`` where
``XXX`` is your architecture number.
If CUDA is installed in a non-standard location and ``cmake`` cannot
find it, you can specify also
``-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda``.

Now, build the Python modules (as above, we assume Cython is installed):

After running ``make -j 2``, you should have the file ``_dynet.so`` in the ``build/python`` folder.

As before, ``cd build/python`` followed by
``python ../../setup.py EIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN build --build-dir=.. --skip-build install --user`` will install the module.

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

Using the GPU from Python
~~~~~~~~~~~~~~~~~~~~~~~~~

The preferred way to make DyNet use the GPU under Python is to import
dynet as usual:

::

    import dynet

Then tell it to use the GPU by using the commandline switch
``--dynet-gpu`` or the GPU switches detailed `here
<commandline.html>`__ when invoking the program. This option lets the
same code work with either the GPU or the CPU version depending on how
it is invoked.

Alternatively, you can also select whether the CPU or GPU should be
used by using ``dynet_config`` module:

::

    import dynet_config
    dynet_config.set_gpu()
    import dynet

This may be useful if you want to decide programmatically whether to
use the CPU or GPU. Importantly, it is not suggested to use ``import _dynet``
any more.
    

Running with MKL
~~~~~~~~~~~~~~~~

If you've built DyNet to use MKL (using ``-DMKL`` or ``-DMKL_ROOT``), Python sometimes has difficulty finding
the MKL shared libraries. You can try setting ``LD_LIBRARY_PATH`` to point to your MKL library directory.
If that doesn't work, try setting the following environment variable (supposing, for example,
your MKL libraries are located at ``/opt/intel/mkl/lib/intel64``):

.. code:: bash

    export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so


