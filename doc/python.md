# Installing the Python DyNet module.

(for instructions on installing on a computer with GPU, see below)

Python bindings to DyNet are currently only supported under python 2.

## TL;DR 
(see below for the details)

```bash
# Installing python DyNet on a machine with python 2.7:

pip install cython # if you don't have it already.
mkdir dynet-base
cd dynet-base
# getting dynet and eigen
git clone https://github.com/clab/dynet.git
hg clone https://bitbucket.org/eigen/eigen
cd dynet
mkdir build
cd build
# without GPU support:
cmake .. -DEIGEN3_INCLUDE_DIR=../eigen -DPYTHON=`which python`
# or with GPU support:
cmake .. -DEIGEN3_INCLUDE_DIR=../eigen -DPYTHON=`which python` -DBACKEND=cuda

make -j 2 # replace 2 with the number of available cores
cd python
python setup.py install  # or `python setup.py install --user` for a user-local install.
```

## Detailed Instructions:
First, get DyNet:

```bash
cd $HOME
mkdir dynet-base
cd dynet-base
git clone https://github.com/clab/dynet.git
cd dynet
git submodule init # To be consistent with DyNet's installation instructions.
git submodule update # To be consistent with DyNet's installation instructions.
```

Then get Eigen:

```bash
cd $HOME
cd dynet-base
hg clone https://bitbucket.org/eigen/eigen/
```

We also need to make sure the `cython` module is installed.
(you can replace `pip` with your favorite package manager, such as `conda`, or install within a virtual environment)
```bash
pip install cython
```

To simplify the following steps, we can set a bash variable to hold where we have saved the main directories of DyNet and Eigen. In case you have gotten DyNet and Eigen differently from the instructions above and saved them in different location(s), these variables will be helpful:

```bash
PATH_TO_DYNET=$HOME/dynet-base/dynet/
PATH_TO_EIGEN=$HOME/dynet-base/eigen/
```

Compile DyNet.

This is pretty much the same process as compiling DyNet, with the addition of the `-DPYTHON=` flag, pointing to the location of your python interpreter.

If boost is installed in a non-standard location, you should add the corresponding flags to the `cmake` commandline,
see the [DyNet installation instructions page](install.md).

```bash
cd $PATH_TO_DYNET
PATH_TO_PYTHON=`which python`
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DPYTHON=$PATH_TO_PYTHON
make -j 2
```

Assuming that the `cmake` command found all the needed libraries and didn't fail, the `make` command will take a while, and compile dynet as well as the python bindings.
You can change `make -j 2` to a higher number, depending on the available cores you want to use while compiling.

You now have a working python binding inside of `build/dynet`.
To verify this is working:

```bash
cd $PATH_TO_DYNET/build/python
```
then, within python:
```bash
import dynet as dy
print dy.__version__
model = dy.Model()
```

In order to install the module so that it is accessible from everywhere in the system, run the following:
```bash
cd $PATH_TO_DYNET/build/python
python setup.py install --user
```

(the `--user` switch will install the module in your local site-packages, and works without root privilages.
 To install the module to the system site-packages (for all users), run `python setup.py install` without this switch)


You should now have a working python binding (the dynet module).

Note however that the installation relies on the compiled dynet library being in `$PATH_TO_DYNET/build/dynet`,
so make sure not to move it from there.

Now, check that everything works:

```bash
# check that it works:
cd $PATH_TO_DYNET
cd pyexamples
python xor.py
python rnnlm.py rnnlm.py
```

Alternatively, if the following script works for you, then your installation is likely to be working:
```
from dynet import *
model = Model()
```

## Installing with GPU support

For installing on a computer with GPU, first install CUDA.
The following instructions assume CUDA is installed.

The installation process is pretty much the same, while adding the `-DBACKEND=cuda` flag to the `cmake` stage:

```bash
cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DPYTHON=$PATH_TO_PYTHON -DBACKEND=cuda
```

(if CUDA is installed in a non-standard location and `cmake` cannot find it, you can specify also `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda`.) 

Now, build the python modules (as above, we assume cython is installed):

After running `make -j 2`, you should have the files `_dynet.so` and `_gdynet.so` in the `build/python` folder.

As before, `cd build/python` followed by `python setup.py install --user` will install the module.

# Using the GPU:

In order to use the GPU support, you can either:

* Use `import _gdynet as dy` instead of `import dynet as dy`
* Or use the commandline switch `--dynet-gpu` or the GPU switches detailed [here](commandline.md) when invoking the program.
