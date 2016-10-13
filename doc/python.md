# Installing the Python DyNet module.

(for instructions on installing on a computer with GPU, see below)

Python bindings to DyNet are currently only supported under python 2.

First, get DYNET:

```bash
cd $HOME
mkdir dynet
git clone https://github.com/clab/dynet.git
cd dynet
git submodule init # To be consistent with DyNet's installation instructions.
git submodule update # To be consistent with DyNet's installation instructions.
```

Then get Eigen:

```bash
cd $HOME
cd dynet
hg clone https://bitbucket.org/eigen/eigen/
```

We also need to make sure the `cython` module is installed.
(you can replace `pip` with your favorite package manager, such as `conda`, or install within a virtual environment)
```bash
pip install cython
```

To simplify the following steps, we can set a bash variable to hold where we have saved the main directories of DyNet and Eigen. In case you have gotten DyNet and Eigen differently from the instructions above and saved them in different location(s), these variables will be helpful:

```bash
PATH_TO_DYNET=$HOME/dynet/dynet/
PATH_TO_EIGEN=$HOME/dynet/eigen/
```

Compile DyNet.
(modify the code below to point to the correct boost location. Note the addition of the -DPYTHON flag.)

```bash
cd $PATH_TO_DYNET
PATH_TO_PYTHON=`which python`
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DBOOST_ROOT=$HOME/.local/boost_1_58_0 -DBoost_NO_BOOST_CMAKE=ON -DPYTHON=$PATH_TO_PYTHON
make -j 2
```

Assuming that the `cmake` command found all the needed libraries and didn't fail, the `make` command will take a while, and compile dynet as well as the python bindings.

You now have a working python binding inside of `build/dynet`.
To verify this is working:

```bash
cd $PATH_TO_DYNET/build/python
```
then, within python:
```bash
import dynet as pc
print pc.__version__
model = pc.Model()
```

In order to install the module so that it is accessible from everywhere, run the following:
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
## Currently unsupported. The GPU support instructions need some revisions.

For installing on a computer with GPU, first install CUDA.
Here, we assume CUDA is installed in `/usr/local/cuda-7.5`

There are two modules, `dynet` which is the regular CPU module, and `gdynet` which is the GPU
module. You can import either of them, these are two independent modules. The GPU support
is incomplete: some operations (i.e. `hubber_distance`) are not available for the GPU.

First step is to build the DyNet modules.
Checkout and go to the `build` directory (same instructions as above). Then:

To build a CPU version on a computer with CUDA:
```bash
cmake .. -DEIGEN3_INCLUDE_DIR=../eigen -DBACKEND=eigen
make -j 4
```

To build a GPU version on a computer with CUDA:
```bash
cmake .. -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.5/
make -j 4
```

Now, build the python modules (as above, we assume cython is installed):

The GPU module (gdynet):
```bash
cd ../dynet
make gdynet.so
make ginstall
```

The CPU module (dynet):
```bash
cd ../dynet
make dynet.so
make install
```

Add the following to your env:
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PATH_TO_DYNET/dynet`

Once both the `dynet` and `gdynet` are installed, run `python ../pyexamples/cpu_vs_gpu.py` for a small timing example.


