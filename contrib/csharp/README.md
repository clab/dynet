# DyNetSharp - C# wrapper for DyNet: The Dynamic Neural Network Toolkit

### Installing the library
##### Prerequisites: 
Before using DyNetSharp, you need to have the Microsoft Visual C++ Redistributable installed on the machine. The Visual Studio installation comes with it by default, otherwise you can download and install it from [here](https://go.microsoft.com/fwlink/?LinkId=746572).
> Note: The library only works with x64 executables.

##### Option 1: Installing from NuGet:
Open the Package-Manger-Console in Visual Studio, and type:
    
    PM> Install-Package DynetSharp
    
> This installs the CPU-only version. If you want to use GPU, you have to compile from source (see below).

##### Option 2: Building and installing from source:

Most of these instructions are copied out of the documentation for compiling DyNet for C++ for windows (https://dynet.readthedocs.io/en/latest/install.html#windows-support), with a few addition for installing with GPU support.

##### Prerequisites: 
- Git for windows (https://git-scm.com/download/win)
- TortoiseHg for windows (https://tortoisehg.bitbucket.io/download/index.html)
- CMake for windows (https://cmake.org/download/) [**If you are compiling with Visual Studio 2019, you have to have cmake 3.14+**]
- [GPU only] Nvidia GPU Toolkit (https://developer.nvidia.com/cuda-downloads)
- [GPU only] Nvidia CuDNN library (https://developer.nvidia.com/rdp/cudnn-download) [this requires creating a free login and signing in].
    * The C# solution assumes that you will extract the library to the path: c:\cudnn\cuda\
    * If you don't, you'll have to modify the solution files [explained below].

> Make sure all the installations add their binaries to the PATH variable

##### Downloading the source code: 
```bash
mkdir eigen
cd eigen
wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
unzip eigen-b2e267dc99d4.zip
cd ..
git clone https://github.com/clab/dynet.git
cd dynet
mkdir build
cd build
```

##### Building DyNet for C++: 
Building for CPU:
```bash
cmake .. -DEIGEN3_INCLUDE_DIR=..\..\eigen -G"Visual Studio 14 2015 Win64"
```
Building for GPU:
```bash
cmake .. -DEIGEN3_INCLUDE_DIR=..\..\eigen -G"Visual Studio 14 2015 Win64" -DBACKEND=cuda -DCUDNN_ROOT=c:\cudnn\cuda
```

If you have a different version of Visual Studio, you just need to modify it, e.g.:
- Visual Studio 2017: "Visual Studio 15 2017 Win64"
- Visual Studio 2019: "Visual Studio 16 2019"

[GPU only] If you saved CuDNN in a different directory, you need to modify that here as well.

This will generate dynet.sln. Simply open this and build all. Note: multi-process functionality is currently not supported in Windows, so the multi-process examples (`*-mp`) will not be included in the generated solution.
For extra speed (approximately 30% gain), make the following two changes to the properties page for the "dynet" project (Click on the "dynet" project in the solution explorer, and click Alt-Enter):
- C/C++ -> Preprocessor -> Preprocessor Definitions
    - Add `__FMA__;` to the beginning.
- C/C++ -> Code Generation -> Floating Point Model
    - Change to `Fast (/fp:fast)`
 > Note: Changing this only changes it in the active configuration (Debug/Release/RelWithDebInfo/MinSizeRel). Make sure you change it in the configuration you are building in.

This completes building DyNet for C++. After that's done, you can move on to building DyNet for C#.

##### Building DyNet for C#: 

> [GPU only] If you are building for GPU, you have to change the configuration file. Navigate to the C# directory: `dynet\contrib\csharp\dynetsharp`.
In the `dynetsharp` sub-directory, there are two `.vcxproj` files. 
Rename the `dynetsharp.vcxproj` file to a `dynetsharp_CPU.vcxproj`, and then rename `dynetsharp_GPU.vcxproj` to `dynetsharp.vcxproj`. 

Open the C# solution, which is located in the following path: `dynet\contrib\csharp\dynetsharp\dynetsharp.sln`.

> [GPU only] If you saved CuDNN in a different directory, you need to modify the paths in the Linker/Include directories in the relevant configurations. 

When building the C# code, the solution might give you an error in the file `devices.h`, saying that the header `<unsupported/Eigen/CXX11/Tensor>` cannot be found - just comment out that line in the header file, it's not needed for the C# wrapper.

Build the dynetsharp solution with the desired configuration, and it will generate the `dynetsharp.dll` file for use directly in your C# program.

> [GPU only] For runtime, make sure that the following DLLs are accessiable via the PATH variable, otherwise the program won't run. (Alternatively, you can just copy them into the C# utility build path):
>&nbsp;&nbsp;&nbsp;&nbsp;`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\*.dll`
>&nbsp;&nbsp;&nbsp;&nbsp;`C:\cudnn\cuda\bin\cudnn*.dll`

### Using DyNetSharp 
The wrapper was built very similarly to the python wrapper, so much of the sample code will look similar. 

##### Importing the project:

```cs
using dynetsharp;
```
A lot of the dynet function are static functions (such as "_average_", "_concatenate_", etc.), and in order to use them you need to call the static class they appear in. In order to make life easier, you can import them so they are directly callable from any context:
```cs
using static dynetsharp.DynetFunctions;
...
Expression e = random_normal(new[] { 10, 10});
```
Or, you can assign the class to a shorter name (e.g., dy):
```cs
using dy = dynetsharp.DynetFunctions;
...
Expression e = dy.random_normal(new[] { 10, 10});
```

##### Initializing DyNet:
Before using dynet in the code, you first need to initialize it. 
There are two ways of initializing. First method is to initialize a ```DynetParams``` object, set the parameters, and call the initialize function:
```cs
DynetParams dp = new DynetParams();
// set random seed to have the same result each time
dp.RandomSeed = 0;
dp.Initialize();
```
Second method is to let dynet parse the arguments given to the main:
```cs
static void Main(string[] args) {
        DynetParams.FromArgs(args).Initialize();
}
```
> Note: The DynetParams.FromArgs function returns a DynetParams object, so you can capture the object and have some of the parameters set from the args, and some in the code.
> All the parameters must be set before the first initialize, otherwise they won't be acknowledged in the current run. 

**Only when using CPU:** DyNet has a dynamic memory pool, and as the computation graph grows, DyNet will automatically increase its memory pool to fit all the computations. In C++ and in Python, that memory is never released while running. In C#, you can set a variable in the ```DynetParams``` called the "MaxMemDescriptor", and whenever you renew the computation graph, if DyNet's memory pool is larger than that number, it will release all the memory and reallocate the original amount of requested memory (default is 512mb).
With this being said, since DyNetSharp allows memory-reallocation, you can update the MemDescriptor/MaxMemDescriptor after initializing, and the changes will go into effect with the next comptuation graph. Usage:
```cs
DynetParams dp = new DynetParams();
dp.Initialize();
...
dp.MaxMemDescriptor = 1024;
dp.UpdateMemDescriptors();
```

##### Using DyNet Sharp with GPU:
If you compiled with the GPU mode, it will by default use 1 GPU. You can customize the GPU usage by specifying it in the `DynetParams` prior to initialization. You can specify which devices you want to load in a few ways. 
```cs
// 3 alternate ways to specify the GPU devices to use, choose one:
// 1] Specifying the quantity:
dp.SetRequestedGPUs(2);
// 2] Boolean array with true/false by index of GPU. So to load GPU:0 and GPU:3
dp.SetGPUMask(new[] { true, false, false, true });
// 3] Specifying the IDs of the devices to load (here, you can specify a CPU as well):
dp.SetDeviceIDs("GPU:0", "CPU", "GPU:1", "GPU:2");
// or as a single string:
dp.SetDeviceIDS("GPU:0,CPU,GPU:1,GPU:2");
```
> Note: You can only specify the requested GPUs with one of the above 3 methods. Calling a second method will nullify the previous call.

Each `Expression`/`Parameter`/`LookupParameter` is stored on a single device. Therefore, when you create a Parameter you can specify which device to store it on. Also, when inputting a new expression, you can specify which device to store it on.
For example:
```cs
LookupParameter lp = model.AddLookupParameter(VOCAB_SIZE, DIM, "GPU:1");
Expression e = dy.input(new[] { 1f, 2f, 3f, 4f }, "GPU:0");
Expression e2 = dy.ones(new[] { 100, 50 }, "CPU");
// You can transfer an expression from one device to the other:
Expression e3 = dy.ToDevice(e, "GPU:1");
// To see the list of available devices:
List<string> allDevices = dy.GetListOfAvailableDevices();
```

##### Creating a new computation graph
There is a single global computation graph that is used at any point. dy.RenewCG() clears the current one and starts a new one:
```cs
dy.RenewCG();
```

DyNet supports computation graph checkpointing, so at any point you can do the following:
```cs
dy.CheckpointCG();
// do some stuff
dy.RevertCG();
```

##### Expressions
Expressions are used as an interface to the various functions that can be used to build DyNet computation graphs.
```cs 
// create a scalar expression.
Expression x = dy.input(5.0f);
```
```cs
// create a vector expression
Expression v = dy.input(new[] { 1.0f, 2.0f, 3.0f });
```
```cs
// create a matrix expression from an array
Expression mat1 = dy.input(new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } });
// Or, using the DyNetSharp Tensor object:
// You can either input a multi dimensional array (float[], float[][], float[][][], etc.) 
Tensor t2 = new Tensor(new[] { 1.0f, 2.0f });
Expression mat2 = dy.inputTensor(t2);
// Or a single float array and the dimensions:
Tensor t3 = new Tensor(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 });
Expression mat3 = dy.inputTensor(t3);
```
 Calculate the value of an expression:
```cs
// This will run the forward step of the neural network.
float[] vec = mat1.VectorValue();
Tensor t = mat1.TensorValue(); 
float[] vec2 = v.VectorValue(); 
// you can only run this if the value is indeed a scalar.
float val = x.ScalarValue(); 
```

##### Create Parameters
Parameters are things need to be trained. In contrast to a system like Torch where computational modules may have their own parameters, in DyNet parameters are just parameters.

```cs
// Parameters are things we tune during training.
// Usually a matrix or a vector.

// First we create a parameter collection and add the parameters to it.
ParameterCollection m =  new ParameterCollection();
Parameter W = m.AddParameters(new[] { 8, 8 }); // an 8x8 matrix
Parameter b = m.AddParameters(new[] { 8 }); // an 8x1 vector
```
There are several ways to initial parameters:
```cs
// Specifiying parameter initialization
float scale = 1, mean = 0, stddev = 1;

// Creates 3x5 matrix filled with 0 (or any other float)
Parameter p1 = m.AddParameters(new[] { 3, 5 }, new ConstInitializer(0));
// Creates 3x5 matrix initialized with U([-scale, scale])
Parameter p2 = m.AddParameters(new[] { 3, 5 }, new UniformInitializer(scale)); 
// Creates 3x5 matrix initialized with N(mean, stddev)
Parameter p3 = m.AddParameters(new[] { 3, 5 }, new NormalInitializer(mean, stddev));
// Creates 5x5 identity matrix
Parameter p4 = m.AddParameters(new[] { 5, 5 }, new IdentityInitializer());
// Creates 3x5 matrix with glorot init
Parameter p5 = m.AddParameters(new[] { 3, 5 }, new GlorotInitializer());
Parameter p6 = m.AddParameters(new[] { 3, 5 }); // By default, it uses the GlorotInitializer.
// Creates 3x5 matrix with he init
Parameter p7 = m.AddParameters(new[] { 3, 5 }, new NormalInitializer(mean, stddev));
// Creates 3x5 matrix from a float array (size is inferred)
Parameter p8 = m.AddParameters(new[] { 3, 5 }, new FromVectorInitializer(Enumerable.Repeat(1.0f, 3 * 5).ToArray()));

// You can use the parameters naturally as expressions in the computation graph, e.g.:
Expression e = p1;
Expression e2 = p1 * dy.input(new[] { 1f, 2f, 3f, 4f, 5f });
```

##### Create LookupParameters
LookupParameters represents a table of parameters. They are used to embed a set of discrete objects (e.g. word embeddings). They can be sparsely updated.
Similar to parameters, but are representing a "lookup table" that maps numbers to vectors. These are often used for things like word embeddings.
```cs
// For example, this will have VOCAB_SIZE rows, each of DIM dimensions.
int VOCAB_SIZE = 100;
int[] DIM = new[] { 10 };
LookupParameter lp = m.AddLookupParameters(VOCAB_SIZE, DIM);

// Create expressions from lookup parameters:
Expression e5  = dy.lookup(lp, 5); // create an Expression from row 5.
e5  = lp[5];        // same
Expression e5c = dy.lookup(lp, 5, fUpdate: false); // as before, but don't update when optimizing.
```
Similar to Parameters, we have several ways to initialize LookupParameters.
```cs
// Creates 3x5 matrix filled with 0 (or any other float)
m.AddLookupParameters(VOCAB_SIZE, DIM, new ConstInitializer(0f));
// Similarly with the rest of the initializers, see the Parameters section for more examples.
```

##### More Expression Manipulation
DyNet provides hundreds of operations on Expressions. The user can manipulate Expressions, or build complex Expression easily.
```cs
// First we create some vector Expressions.
Expression e1 = dy.input(new[] { 1f, 2f, 3f, 4f });
Expression e2 = dy.input(new[] { 5f, 6f, 7f, 8f });

// Concatenate list of expressions to a single expression.
Expression eCombined = dy.concatenate(e1, e2);
// Concatenate columns of a list of expression to a single expression
Expression eColCombined = dy.concatenate_cols(e1, e2);

// Basic Math Operations

Expression e, e_;
// Add
e = e1 + e2; // Element-wise addition
// Minus
e = e2 - e1; // Element-wise minus
// Negative
e = -e1; // Should be [-1.0, -2.0, -3.0, -4.0]

//  Multiply
e = e1 * dy.transpose(e1); // It's Matrix multiplication
// Dot product
e = dy.dot_product(e1, e2); // dot product = sum(component-wise multiply)

// Component-wise multiply
e = dy.cmult(e1, e2);
// Component-wise division
e = dy.cdiv(e1, e2);

// Column-wise addition
//  x:  An MxN matrix
//  y:  A length M vector
Expression mat = dy.colwise_add(x, y);

// Useful math operations
// abs()
e = dy.abs(e1);
// cube() - Elementwise cubic
e = dy.cube(e1);
// exp()
e = dy.exp(e1);
// pow() - For each element in e1, calculate e1^{y}
e = dy.pow(e1, dy.input(2));
e_ = dy.square(e1);
// min() - Calculate an output where the ith element is min(x_i, y_i)
e = dy.min(e1, e2);
// max() - Calculate an output where the ith element is max(x_i, y_i)
e = dy.max(e1, e2);
// sin()
e = dy.sin(e1);
// cos()
e = dy.cos(e1);
// tan()
e = dy.tan(e1);
// asin();
e = dy.asin(e1);
// acos()
e = dy.acos(e1);
// atan()
e = dy.atan(e1);
// sinh()
e = dy.sinh(e1);
// cosh()
e = dy.cosh(e1);
// tanh()
e = dy.tanh(e1);
// asinh()
e = dy.asinh(e1);
// acosh()
e = dy.acosh(e1);
// atanh
e = dy.atanh(e1);
// square()
e = dy.square(e1);
// sqrt()
e = dy.sqrt(e1);
```
##### Matrix manipulation
```cs
// Reshape
e = dy.reshape(e1, new[] { 2, 2 }); // Col major
// Transpose
e = dy.transpose(e1);
Console.WriteLine("e1 dimension: " + string.Join(", ", e1.Shape()));
Console.WriteLine("e1 transpose dimension: " + string.Join(", ", e.Shape()));

// inverse
e = dy.inverse(dy.inputTensor(new Tensor(new[] { 1f, 3f, 3f, 1f }, new[] { 2, 2 })));

// logdet
e = dy.logdet(dy.inputTensor(new Tensor(new[] { 1f, 0f, 0f, 2f }, new[] { 2, 2 })));

// trace_of_product
Expression diag_12 = dy.inputTensor(new Tensor(new[] { 1f, 0f, 0f, 2f }, new[] { 2, 2 }));
e = dy.trace_of_product(diag_12, diag_12);

// circ_conv
Expression sig_1 = dy.input(new[] { 1f, 2f, 1f, 0f });
Expression sig_2 = dy.input(new[] { 0f, 1f, 1f, 1f });
e = dy.circ_conv(sig_1, sig_2);

// circ_corr
e = dy.circ_corr(sig_1, sig_2);
```

##### Other Per-element unary functions.
```cs
// erf() - Elementwise calculation of the Gaussian error function
e = dy.erf(e1);
// log()
e = dy.log(e1);
// log_sigmoid()
e = dy.log_sigmoid(e1);
// lgamma() - Definition of gamma function ca be found here:
// https://en.wikipedia.org/wiki/Gamma_function
e = dy.lgamma(e1);
e_ = dy.log(dy.input(new[] { 1f, 1f, 2f, 6f }));
// sigmoid()
e = dy.logistic(e1); // Sigmoid(x)
// rectify() - Rectifier (or ReLU, Rectified Linear Unit)
e = dy.rectify(e1); // Relu (= max(x,0))
// elu() - Exponential Linear Unit (ELU)
// Definition can be found here: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELUs
e = dy.elu(e1);
// selu() - Scaled Exponential Linear Unit (SELU)
// Definition can be found here: https://arxiv.org/abs/1706.02515
e = dy.selu(e1);
// silu() - Sigmoid Linear Unit / sigmoid-weighted linear unit - SILU / SiL / Swish
// Definition can be found here: https://openreview.net/pdf?id=Bk0MRI5lg
e = dy.silu(e1);
// sparsemax() - Similar to softmax, but induces sparse solutions where 
// most of the vector elements are zero.
e = dy.sparsemax(e1);

// softsign()
e = dy.softsign(e1); // x/(1+|x|)
// softmax()
e = dy.softmax(e1); 

// log_softmax
// logsoftmax = logits - log(reduce_sum(exp(logits), dim))
// restrict is a set of indices. if not empty, only entries in restrict are part of softmax computation, others get -inf.
Expression e_log_softmax = dy.log_softmax(e1);
// constrained_softmax()
// similar to softmax, but defines upper bounds for the resulting probabilities. 
e = dy.constrained_softmax(e1, dy.input(new[] { 0.01f, 0.05f, 0.10f, 0.55f }));
```

##### Picking values from vector expressions
```cs
int k = 1, v = 3;
// Pick one element from a vector or matrix
e = dy.pick(e1, k);
e = e1[k]; // equivalent
// Picking a range
mat = dy.pickrange(e1, k, v);

// pickneglogsoftmax
// which is equivalent to: dy.pick(-dy.log(dy.softmax(e1)), k);
e = dy.pickneglogsoftmax(e1, k);
e_ = dy.pick(-dy.log(dy.softmax(e1)), k);
```

##### Selecting vectors from matrix Expressions
```cs
// select_rows [works similar to pickrange]
e = dy.select_rows(mat1, new[] { 0, 1 });
// select_cols
e = dy.select_cols(mat1, new[] { 0 });
```

##### Expressions concatenation & other useful manipuulations
```cs
// This performs an elementwise sum over all the expressions included.
// All expressions should have the same dimension.
e = dy.esum(e1, e2); // for people used to python
e = dy.sum(e1, e2); 
// which is equivalent to:
e_ = e1 + e2;

// This performs an elementwise average over all the expressions included.
// All expressions should have the same dimension.
e = dy.average(e1, e2);
// which is equivalent to:
e_ = (e1 + e2) / 2;

// Concate vectors/matrix column-wise
// All expressions should have the same dimension.
// e1, e2,.. are column vectors. return a matrix.
e = dy.concatenate_cols(e1, e2);

// Concate vectors/matrix
// All expressions should have the same dimension.
// e1, e2,.. are column vectors. return a matrix. 
e = dy.concatenate(e1, e2);

// affine transform
Expression e0 = dy.input(new[] { -1f, 0f });
e = dy.affine_transform(e1, e, e0);

// sum_elems
// Sum all elements
e = dy.sum_elems(e);
```
And many more. See the regular documentation for more possible manipulations.

### DyNet in Neural Networks
This part contains Neural Networks related issues.

##### Noise and Dropout Expressions
```cs
// Add a noise to each element from a gausian distribution
// with standard-dev = stddev
float stddev = 0.1f;
e = dy.noise(e1, stddev);

// Apply dropout to the input expression
// There are two kinds of dropout methods 
// (http://cs231n.github.io/neural-networks-2)
// Dynet implement the Inverted dropout where dropout with prob p 
// and scaling others by 1/p at training time, and do not need 
// to do anything at test time. 
float p = 0.5f;
e = dy.dropout(e1, p); // apply dropout with probability p 
// If we set p=1, everything will be dropped out
e = dy.dropout(e1, 1);
// If we set p=0, everything will be kept
e = dy.dropout(e1, 0);
```
##### Loss Functions
```cs
// DyNet provides several ways to calculate "distance"
// between two expressions of the same dimension
// This is square_distance, defined as
// sum(square of(e1-e2)) for all elements
// in e1 and e2.
// Here e1 is a vector of [1,2,3,4]
// And e2 is a vector of [5,6,7,8]
// The square distance is sum((5-1)^2 + (6-2)^2+...)
e = dy.squared_distance(e1, e2);

// This is the l1_distance, defined as 
// sum (abs(e1-e2)) for all elements in
// e1 and e2.
e = dy.l1_distance(e1, e2);

// This is the huber_distance, definition 
// found here. (https://en.wikipedia.org/wiki/Huber_loss)
// The default threhold (delta) is 1.345.
// Here e1 is a vector of [1,2,3,4]
// And e2 is a vector of [5,6,7,8]
// because for each pair-wised element in
// e1 and e2, the abs(e1-e2)=4>delta=1.345,
// so the output is sum(delta*(abs(4)-1/2*delta))
e = dy.huber_distance(e1, e2, c: 1.345);

// Binary logistic loss function
// This is similar to cross entropy loss function
// e1 must be a vector that takes values between 0 and 1
// ty must be a vector that takes values between 0 and 1
e = -(ty * dy.log(e1) + (1 - ty) * dy.log(1 - e1));
ty = dy.input(new[] { 0f, 0.5f, 0.5f, 1f });
Expression e_scale = dy.inputVector(4);
e_scale.SetValue(new[] { 0.5f, 0.5f, 0.5f, 0.5f });
e = dy.binary_log_loss(e_scale, ty);
// The binary_log_loss is equivalent to the following:
Expression e_equl = -(dy.dot_product(ty, dy.log(e_scale)) + dy.dot_product((dy.input(new[] { 1f, 1f, 1f, 1f }) - ty), dy.log(dy.input(new[] { 1f, 1f, 1f, 1f }) - e_scale)));

// pairwise_rank_loss
// e1 is row vector or scalar
// e2 is row vector or scalar
// m is number
// e = max(0, m - (e1 - e2))
e = dy.pairwise_rank_loss(dy.transpose(e1), dy.transpose(e2), m: 1.0f); // Row vector needed, so we transpose the vector.

// and many more
```
Other topics that the documentation can be found for C++/python:
- Convolutions
- Backpropagation and Gradients
- Normalization 

### Write your own Neural Networks
Now that you have a basic idea about APIs, you can try to write simple Neural Networks of your own.

On the GitHub repository, we have a few examples (more to come) of C# implementations of common neural network challenges (XOR, RNNLM, and more);
