# cnn
C++ neural network library

### Important: Eigen version requirement

You need the [development version of the Eigen library](https://bitbucket.org/eigen/eigen) for this software to function. **If you use any of the released versions, you may get assertion failures or compile errors.**

### Building

First you need to fetch the dependent libraries

    git submodule init
    git submodule update

In `src`, you need to first use [`cmake`](http://www.cmake.org/) to generate the makefiles

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen

Then to compile, run

    make -j 2

To see that things have built properly, you can run

    ./examples/xor

which will train a multilayer perceptron to predict the xor function.

#### Building without Eigen installed

If you don't have Eigen installed, the instructions below will fetch and compile
both `Eigen` and `cnn`. Eigen does not have to be compiled, so “installing” it is easy.
        
    git clone https://github.com/clab/cnn.git
    hg clone https://bitbucket.org/eigen/eigen/

    cd cnn/
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=../eigen
    make -j 2

#### Building with GPU Support

`cnn` supports running programs on GPUs with CUDA. If you have CUDA installed, you
can build cnn to be run on GPUs by adding `-DBACKEND=cuda` to your cmake options
as follows:

    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DBACKEND=cuda

#### Debugging build problems

If you want to see the compile commands that are used, you can run

    make VERBOSE=1

### Command line options

All programs using cnn have a few command line options. These must be specified at the
very beginning of the command line, before other options.

* `--cnn-mem NUMBER`: cnn runs by default with 512MB of memory each for the forward and
  backward steps, as well as parameter storage. You will often want to increase this amount.
  By setting NUMBER here, cnn will allocate more memory. Note that it will allocate 3 times
  more memory than the number specified here, so if you want to use 3GB, specify "1024".
* `--cnn-l2 NUMBER`: Specifies the level of l2 regularization to use.
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
cg.PrintGraphviz();

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
