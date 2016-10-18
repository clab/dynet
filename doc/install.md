# Building/Installing
How to build DyNet and link it with your programs

## Prerequisites

DyNet relies on a number of external libraries including Boost, cmake, Eigen, and mercurial (to install Eigen).
Boost, cmake, and mercurial can be installed from standard repositories, for example on Ubuntu linux:

    sudo apt-get install libboost-all-dev cmake mercurial

To compile DyNet you also need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen). **If you use any of the released versions, you may get assertion failures or compile errors.** If you don't have Eigen installed already, you can get it easily using the following command:

    hg clone https://bitbucket.org/eigen/eigen/

## Building

To get and build DyNet, clone the repository

    git clone https://github.com/clab/dynet.git

then enter the directory and use [`cmake`](http://www.cmake.org/) to generate the makefiles

    cd dynet
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then compile, where "2" can be replaced by the number of cores on your machine

    make -j 2

To see that things have built properly, you can run

    ./examples/xor

which will train a multilayer perceptron to predict the xor function.

## Compiling/linking External Programs

When you want to use DyNet in an external program, you will need to add the `dynet`
directory to the compile path:

    -I/path/to/dynet

and link with the dynet library:

    -L/path/to/dynet/build/dynet -ldynet

## Debugging build problems

If you have a build problem and want to debug, please run 

    make clean
    make VERBOSE=1 &> make.log

then examine the commands in the `make.log` file to see if anything looks fishy. If
you would like help, send this `make.log` file via the "Issues" tab on github, or to
the dynet-users mailing list.

## Build options

### GPU (CUDA) support

`dynet` supports running programs on GPUs with CUDA. If you have CUDA installed, you
can build DyNet with GPU support by adding `-DBACKEND=cuda` to your cmake options.
This will result in three libraries named "libdynet," "libgdynet," and "libdynetcuda" being 
created. When you want to run a program on CPU, you can link to the "libdynet" library as
shown above. When you want to run a program on GPU, you can link to the "libgdynet" and 
"libdynetcuda" libraries.

    -L/path/to/dynet/build/dynet -lgdynet -ldynetcuda

(Eventually you will be able to use a single library to run on either CPU or GPU, but this is
not fully implemented yet.)

### Non-standard Boost location

`dynet` supports boost, and will find it if it is in the standard location. If boost is
in a non-standard location, say `$HOME/boost`, you can specify the location by adding
the following to your cmake options:

    -DBOOST_ROOT:PATHNAME=$HOME/boost -DBoost_LIBRARY_DIRS:FILEPATH=$HOME/boost/lib
    -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE

Note that you will also have to set your `LD_LIBRARY_PATH` to point to the `boost/lib`
directory.

### Building for Windows

DYNET has been tested to build in Windows using Microsoft Visual Studio 2015. You may be able to build with MSVC 2013 by slightly modifying the instructions below.

First, install Eigen following the above instructions.

Second, install [Boost](http://www.boost.org/) for your compiler and platform. Follow the instructions for compiling Boost or just download the already-compiled binaries.

To generate the MSVC solution and project files, run [cmake](http://www.cmake.org), pointing it to the location you installed Eigen and Boost (for example, at c:\libs\Eigen and c:\libs\boost_1_61_0):

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=c:\libs\Eigen -DBOOST_ROOT=c:\libs\boost_1_61_0 -DBOOST_LIBRARYDIR=c:\libs\boost_1_61_0\lib64-msvc-14.0 -DBoost_NO_BOOST_CMAKE=ON -G"Visual Studio 14 2015 Win64"

This will generate dynet.sln and a bunch of \*.vcxproj files (one for the DYNET library, and one per example). You should be able to just open dynet.sln and build all. **Note: multi-process functionality is currently not supported in Windows, so you will not be able to build rnnlm-mp. Go to build->Configuration Manager and uncheck the box next to this project**. 

The Windows build also supports CUDA. The latest (development) version of Eigen has some code that causes problems with the CUDA compiler. These issue change as Eigen is developed. Currently, the following three changes are needed in Eigen when compiling with CUDA support:

- block.h: add `#ifndef __CUDACC__` / `#endif` around `EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Block)`
- ref.h: add `#ifndef __CUDACC__ / #endif` around `EIGEN_INHERIT_ASSIGNMENT_OPERATORS(RefBase)`
- TensorRandom.h: Change `uint` to `unsigned int` and `SYSTEMTIME st` to immediately above `GetSystemTime(&st)`
- TensorDeviceCuda.h: Change `sleep(1)` to `Sleep(1000)`

### MKL support

DyNet can leverage Intel's MKL library to speed up computation on the CPU. As an example, we've seen 3x speedup in seq2seq training when using MKL. To use MKL, include the following cmake option: 

    -DMKL=TRUE

If cmake is unable to find MKL automatically, try setting `MKL_ROOT`, such as

    -DMKL_ROOT="/path/to/MKL"

If either MKL or MKL_ROOT are set, CMake will look for MKL. Note, MKL has only been tested in Windows.



