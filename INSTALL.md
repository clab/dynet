# Installing the pyCNN module.

First, get CNN and Eigen:

```bash
cd $HOME
git clone https://github.com/clab/cnn.git
cd cnn
git submodule init # To be consistent with CNN's installation instructions.
git submodule update # To be consistent with CNN's installation instructions.
# hg clone https://bitbucket.org/eigen/eigen/ # Latest version (17.03.16) of Eigen fails to compile.
wget u.cs.biu.ac.il/~yogo/eigen.tgz
tar zxvf eigen.tgz # or "dtrx eigen.tgz" if you have dtrx installed.
```

Compile CNN.
(modify the code below to point to the correct boost location)

```bash
cd $HOME/cnn/
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../eigen -DBOOST_ROOT=$HOME/.local/boost_1_58_0 -DBoost_NO_BOOST_CMAKE=ON
make -j 2
```

Now that CNN is compiled, we need to compile the pycnn module.
This requires having cython installed.
If you don't have cython, it can be installed with either `pip install cython` or better yet `conda install cython`.

```bash
pip2 install cython --user
cd $HOME/cnn/pycnn
make
make install
```

We are almost there. 
We need to tell the environment where to find the compiled cnn shared library.
The pyCNN's `make` fetched a copy of `libcnn_shared.so` and put it in the `pycnn` lib.

Add the following line to your profile (`.zshrc` or `.bashrc`), change
according to your installation location.

```bash
export LD_LIBRARY_PATH=$HOME/cnn/pycnn
```

Now, check that everything works:

```bash
# check that it works:
cd $HOME/cnn
cd pyexamples
python2 xor.py
python2 rnnlm.py rnnlm.py
```

Alternatively, if the following script works for you, then your installation is likely to be working:
```
from pycnn import *
model = Model()
```
