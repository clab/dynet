# cnn
C++ neural network library

cnn is a neural network library that is written in C++ (with bindings in Python). It is designed to be efficient when run on either CPU or GPU, and works well with networks that have dynamic structures that change for every training instance. Read the instructions below to get started, and feel free to contact the [cnn-users group](https://groups.google.com/forum/#!forum/cnn-users) group or [github page](http://github.com/clab/cnn) with any questions, issues, or contributions.

### Building

(for how to use the python bindings, see `PYINSTALL.md`)

Before compiling cnn, you need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) for this software to function. **If you use any of the released versions, you may get assertion failures or compile errors.**

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then compile, where "2" can be replaced by the number of cores on your machine

    make -j 2

To see that things have built properly, you can run

    ./examples/xor

which will train a multilayer perceptron to predict the xor function.

#### Build options

##### Building without Eigen installed

If you don't have Eigen installed, the instructions below will fetch and compile
both `Eigen` and `cnn`. Eigen does not have to be compiled, so “installing” it is easy.
        
    git clone https://github.com/clab/cnn.git
    hg clone https://bitbucket.org/eigen/eigen/

    cd cnn/
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=../eigen
    make -j 2

##### GPU (CUDA) support

`cnn` supports running programs on GPUs with CUDA. If you have CUDA installed, you
can build cnn with GPU support by adding `-DBACKEND=cuda` to your cmake options.
This will result in two libraries named "libcnn" and "libgcnn" being created. When
you want to run a program on CPU, you can link to the "libcnn" library, and when
you want to run a program on GPU, you can link to the "libgcnn" library. (Eventually
you will be able to use a single library to run on either CPU or GPU, but this is
not fully implemented yet.)

##### Non-standard Boost location

`cnn` supports boost, and will find it if it is in the standard location. If boost is
in a non-standard location, say `$HOME/boost`, you can specify the location by adding
the following to your cmake options:

    -DBOOST_ROOT:PATHNAME=$HOME/boost -DBoost_LIBRARY_DIRS:FILEPATH=$HOME/boost/lib
    -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE

Note that you will also have to set your `LD_LIBRARY_PATH` to point to the `boost/lib`
directory.

#### Building for Windows

CNN has been tested to build in Windows using Microsoft Visual Studio 2015. You may be able to build with MSVC 2013 by slightly modifying the instructions below.

First, install Eigen following the above instructions.

Second, install [Boost](http://www.boost.org/) for your compiler and platform. Follow the instructions for compiling Boost or just download the already-compiled binaries.

To generate the MSVC solution and project files, run [cmake](http://www.cmake.org), pointing it to the location you installed Eigen and Boost (for example, at c:\libs\Eigen and c:\libs\boost_1_61_0):

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=c:\libs\Eigen -DBOOST_ROOT=c:\libs\boost_1_61_0 -DBOOST_LIBRARYDIR=c:\libs\boost_1_61_0\lib64-msvc-14.0 -DBoost_NO_BOOST_CMAKE=ON -G"Visual Studio 14 2015 Win64"

This will generate cnn.sln and a bunch of \*.vcxproj files (one for the CNN library, and one per example). You should be able to just open cnn.sln and build all. **Note: multi-process functionality is currently not supported in Windows, so you will not be able to build rnnlm-mp. Go to build->Configuration Manager and uncheck the box next to this project**

#### Debugging build problems

If you want to see the compile commands that are used, you can run

    make VERBOSE=1

### Command line options

All programs using cnn have a few command line options. These must be specified at the
very beginning of the command line, before other options.

* `--cnn-mem NUMBER`: cnn runs by default with 512MB of memory each for the forward and
  backward steps, as well as parameter storage. You will often want to increase this amount.
  By setting NUMBER here, cnn will allocate more memory. Note that you can also individually
  set the amount of memory for forward calculation, backward calculation, and parameters
  by using comma separated variables `--cnn-mem FOR,BACK,PARAM`. This is useful if, for
  example, you are performing testing and don't need to allocate any memory for backward
  calculation.
* `--cnn-l2 NUMBER`: Specifies the level of l2 regularization to use (default 1e-6).
* `--cnn-gpus NUMBER`: Specify how many GPUs you want to use, if cnn is compiled with CUDA.
  Currently, only one GPU is supported.
* `--cnn-gpu-ids X,Y,Z`: Specify the GPUs that you want to use by device ID. Currently only
  one GPU is supported, but if you use this command you can select which one to use.

### Creating your own models

An illustation of how models are trained (for a simple logistic regression model) is below:

```c++
// *** First, we set up the structure of the model
// Create a model, and an SGD trainer to update its parameters.
Model mod;
SimpleSGDTrainer sgd(&mod);
// Create a "computation graph," which will define the flow of information.
ComputationGraph cg;
// Initialize a 1x3 parameter vector, and add the parameters to be part of the
// computation graph.
Expression W = parameter(cg, mod.add_parameters({1, 3}));
// Create variables defining the input and output of the regression, and load them
// into the computation graph. Note that we don't need to set concrete values yet.
vector<cnn::real> x_values(3);
Expression x = input(cg, {3}, &x_values);
cnn::real y_value;
Expression y = input(cg, &y_value);
// Next, set up the structure to multiply the input by the weight vector,  then run
// the output of this through a logistic sigmoid function (logistic regression).
Expression y_pred = logistic(W*x);
// Finally, we create a function to calculate the loss. The model will be optimized
// to minimize the value of the final function in the computation graph.
Expression l = binary_log_loss(y_pred, y);
// We are now done setting up the graph, and we can print out its structure:
cg.print_graphviz();

// *** Now, we perform a parameter update for a single example.
// Set the input/output to the values specified by the training data:
x_values = {0.5, 0.3, 0.7};
y_value = 1.0;
// "forward" propagates values forward through the computation graph, and returns
// the loss.
cnn::real loss = as_scalar(cg.forward());
// "backward" performs back-propagation, and accumulates the gradients of the
// parameters within the "Model" data structure.
cg.backward();
// "sgd.update" updates parameters of the model that was passed to its constructor.
// Here 1.0 is the scaling factor that allows us to control the size of the update.
sgd.update(1.0);
```

Note that this very simple example that doesn't cover things like memory initialization, reading/writing models, recurrent/LSTM networks, or adding biases to functions. The best way to get an idea of how to use cnn for real is to look in the `example` directory, particularly starting with the simplest `xor` example.
