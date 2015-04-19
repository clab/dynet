# cnn
C++ neural network library

#### Building

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles

    cmake . -DEIGEN3_INCLUDE_DIR=/Users/cdyer/software/eigen-eigen-36fd1ba04c12

Then to compile, run

    make -j 2
    make test

If you want to see the compile commands that are used, you can run

    make VERBOSE=1

