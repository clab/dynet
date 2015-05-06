# cnn
C++ neural network library

#### Getting started

You need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) for this software to function properly. If you use the current stable release, you will get an error like the following:

    Assertion failed: (false && "heap allocation is forbidden (EIGEN_NO_MALLOC is defined)"), function check_that_malloc_is_allowed, file /Users/cdyer/software/eigen-eigen-10219c95fe65/Eigen/src/Core/util/Memory.h, line 188.

#### Building

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then to compile, run

    make -j 2

To see that things have built properly, you can run

    ./examples/xor

which will train a multilayer perceptron to predict the xor function.

#### Debugging

If you want to see the compile commands that are used, you can run

    make VERBOSE=1

