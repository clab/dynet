# Installing the pyCNN module.

First, get CNN and Eigen:

```bash
mkdir cnn
cd cnn
git clone https://github.com/yoavg/cnn.git
hg clone https://bitbucket.org/eigen/eigen/
```

Compile CNN.
Note that we are currently relying on the CNN code that is in the
`pycnn` branch (it has a few modifications needed for pyCNN to work), 
but at some point this will be merged into cnn proper.
(modify the code below to point to the correct boost location)

```bash
cd cnn
git checkout pycnn # branch
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=../eigen -DBOOST_ROOT=/home/yogo/.local/boost_1_58_0 -DBoost_NO_BOOST_CMAKE=ON
make -j 2
```

Now that CNN is compiled, we need to compile the pycnn module.
This requires having cython installed.
If you don't have cython, it can be installed with either `pip install cython` or better yet `conda install cython`.

```bash
pip2 install cython --user
cd ../pycnn
make
make install
```

We are almost there. 
We need to tell the environment where to find the compiled cnn shared library.
The pyCNN's `make` fetched a copy of `libcnn_shared.so` and put it in the `pycnn` lib.

Add the following line to your profile (`.zshrc` or `.bashrc`), change
according to your installation location.

```bash
export LD_LIBRARY_PATH=/home/yogo/cnn/cnn/pycnn
```

Now, check that everything works:

```bash
# check that it works:
cd ..
cd pyexamples
python2 xor.py
python2 rnnlm.py rnnlm.py
```

