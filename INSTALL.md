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
tar zxvf eigen.tgz # or "dtrx eigen.tgz" if you have dtrx installed, e.g. with "sudo apt-get install dtrx"
```

To simplify the following steps, we can set a bash variable to hold where we have saved the main directories of `cnn` and `eigen`. In case you have gotten `ccn` and `eigen` differently from the instructions above and saved them in different location(s), these variables will be helpful:

```bash
PARENT_DIR_OF_CNN=$HOME 
PATH_TO_EIGEN=$HOME/cnn/eigen
```

Compile CNN.
(modify the code below to point to the correct boost location)

```bash
cd PARENT_DIR_OF_CNN/cnn
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DBOOST_ROOT=$HOME/.local/boost_1_58_0 -DBoost_NO_BOOST_CMAKE=ON
make -j 2
```

Now that CNN is compiled, we need to compile the pycnn module.
This requires having cython installed.
If you don't have cython, it can be installed with either `pip install cython` or better yet `conda install cython`.

```bash
pip2 install cython --user
```

Customize the `setup.py` to include (i) the parent directory where the main `cnn` directory is saved and (ii) the path to the main `eigen` directy:

```bash
cd $PARENT_DIR_OF_CNN/cnn/pycnn
sed -i  "s|..\/..\/cnn\/|$PARENT_DIR_OF_CNN|g" setup.py 
sed -i  "s|..\/..\/eigen\/|$PATH_TO_EIGEN|g" setup.py
make
make install
```

We are almost there. 
We need to tell the environment where to find the compiled cnn shared library.
The pyCNN's `make` fetched a copy of `libcnn_shared.so` and put it in the `pycnn` lib.

Add the following line to your profile (`.zshrc` or `.bashrc`), change
according to your installation location.

```bash
export LD_LIBRARY_PATH=$PARENT_DIR_OF_CNN/cnn/pycnn
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
