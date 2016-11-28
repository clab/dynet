Building/Installing
===================

How to build DyNet and link it with your programs

Prerequisites
-------------

DyNet relies on a number of external libraries including Boost, CMake,
Eigen, and Mercurial (to install Eigen). Boost, CMake, and Mercurial can
be installed from standard repositories, for example on Ubuntu Linux:

::

    sudo apt-get install libboost-all-dev cmake mercurial

Or on OSX, first make sure the Apple Command Line Tools are installed, then
get Boost, CMake, and Mercurial with either homebrew or macports:

::

    xcode-select --install
    brew install boost cmake hg
    sudo port install boost cmake mercurial

To compile DyNet you also need the `development version of the Eigen
library <https://bitbucket.org/eigen/eigen>`__. **If you use any of the
released versions, you may get assertion failures or compile errors.**
If you don't have Eigen installed already, you can get it easily using
the following command:

::

    hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb
    
The `-r NUM` specified a revision number that is known to work.
Adventurous users can remove it and use the very latest version, at the risk of the code breaking / not compiling.

Building
--------

To get and build DyNet, clone the repository

::

    git clone https://github.com/clab/dynet.git

then enter the directory and use ```cmake`` <http://www.cmake.org/>`__
to generate the makefiles

::

    cd dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then compile, where "2" can be replaced by the number of cores on your
machine

::

    make -j 2

To see that things have built properly, you can run

::

    ./examples/xor

which will train a multilayer perceptron to predict the xor function.

Compiling/linking external programs
-----------------------------------

When you want to use DyNet in an external program, you will need to add
the ``dynet`` directory to the compile path:

::

    -I/path/to/dynet

and link with the DyNet library:

::

    -L/path/to/dynet/build/dynet -ldynet

Debugging build problems
------------------------

If you have a build problem and want to debug, please run

::

    make clean
    make VERBOSE=1 &> make.log

then examine the commands in the ``make.log`` file to see if anything
looks fishy. If you would like help, send this ``make.log`` file via the
"Issues" tab on GitHub, or to the dynet-users mailing list.


GPU/MKL support and build options
---------------------------------

GPU (CUDA) support
~~~~~~~~~~~~~~~~~~

``dynet`` supports running programs on GPUs with CUDA. If you have CUDA
installed, you can build DyNet with GPU support by adding
``-DBACKEND=cuda`` to your cmake options. This will result in three
libraries named "libdynet," "libgdynet," and "libdynetcuda" being
created. When you want to run a program on CPU, you can link to the
"libdynet" library as shown above. When you want to run a program on
GPU, you can link to the "libgdynet" and "libdynetcuda" libraries.

::

    -L/path/to/dynet/build/dynet -lgdynet -ldynetcuda

(Eventually you will be able to use a single library to run on either
CPU or GPU, but this is not fully implemented yet.)


MKL support
~~~~~~~~~~~

DyNet can leverage Intel's MKL library to speed up computation on the CPU.
As an example, we've seen 3x speedup in seq2seq training when using MKL. To use MKL, include the following cmake option:

::

    -DMKL=TRUE

If CMake is unable to find MKL automatically, try setting `MKL_ROOT`, such as

::

    -DMKL_ROOT="/path/to/MKL"

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

Non-standard Boost location
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dynet`` requires Boost, and will find it if it is in the standard
location. If Boost is in a non-standard location, say ``$HOME/boost``,
you can specify the location by adding the following to your CMake
options:

::

    -DBOOST_ROOT:PATHNAME=$HOME/boost -DBoost_LIBRARY_DIRS:FILEPATH=$HOME/boost/lib
    -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE

Note that you will also have to set your ``LD_LIBRARY_PATH`` to point to
the ``boost/lib`` directory.
Note also that Boost must be compiled with the same compiler version as
you are using to compile DyNet.

Building for Windows
~~~~~~~~~~~~~~~~~~~~

DyNet has been tested to build in Windows using Microsoft Visual Studio
2015. You may be able to build with MSVC 2013 by slightly modifying the
instructions below.

First, install Eigen following the above instructions.

Second, install `Boost <http://www.boost.org/>`__ for your compiler and
platform. Follow the instructions for compiling Boost or just download
the already-compiled binaries.

To generate the MSVC solution and project files, run
`cmake <http://www.cmake.org>`__, pointing it to the location you
installed Eigen and Boost (for example, at c:\\libs\\Eigen and c:\\libs\\boost_1_61_0):

::

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=c:\libs\Eigen -DBOOST_ROOT=c:\libs\boost_1_61_0 -DBOOST_LIBRARYDIR=c:\libs\boost_1_61_0\lib64-msvc-14.0 -DBoost_NO_BOOST_CMAKE=ON -G"Visual Studio 14 2015 Win64"

This will generate `dynet.sln` and a bunch of `*.vcxproj` files (one for
the DyNet library, and one per example). You should be able to just open
`dynet.sln` and build all. **Note: multi-process functionality is
currently not supported in Windows, so the multi-process examples (`*-mp`) will not be included
in the generated solution**

The Windows build also supports CUDA with the latest version of Eigen (as of Oct 28, 2016), with the following code change: 

- TensorDeviceCuda.h: Change `sleep(1)` to `Sleep(1000)`

