#!/bin/bash
set -xe

# Boost, Python packages
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get -qq update
  sudo apt-get install -y gcc-4.8 g++-4.8 libssl-dev
  PYTHON_PACKAGES="numpy twine auditwheel cython"
  if [[ "$PYTHON_INSTALL" == manual ]]; then
    sudo apt-get install -y --allow-unauthenticated libboost-filesystem1.55-dev libboost-program-options1.55-dev libboost-serialization1.55-dev libboost-test1.55-dev libboost-regex1.55-dev
    sudo -H pip install -U $PYTHON_PACKAGES
  else
    pip install -U pip
    pip install --prefer-binary cryptography
    pip install -U $PYTHON_PACKAGES
  fi
else
  brew update
  # Install Miniconda
  export MINICONDA_OS_NAME=MacOSX MINICONDA_ARCH=x86_64
  wget "https://repo.continuum.io/miniconda/Miniconda3-latest-$MINICONDA_OS_NAME-$MINICONDA_ARCH.sh" -O miniconda.sh
  bash miniconda.sh -b -p miniconda
  export PATH="$PWD/miniconda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda create -c conda-forge -q -n "$PYVER" python="$PYVER" numpy cython
  # Useful for debugging any issues with conda
  conda info -a
  source activate "$PYVER"
  pip install twine
fi

# CUDA
if [[ "$BACKEND" == cuda ]]; then
  CUDA_VERSION_MAJOR="8" CUDA_VERSION_MINOR="0"
  CUDA_PKG_LONGVERSION="${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}.61-1"
  CUDA_PKG_VERSION="${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"
  CUDA_REPO_PKG=cuda-repo-ubuntu1404_${CUDA_PKG_LONGVERSION}_amd64.deb
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
  sudo dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG
  sudo apt-get -y update
  sudo apt-get install -y --no-install-recommends cuda-drivers cuda-core-$CUDA_PKG_VERSION cuda-cudart-dev-$CUDA_PKG_VERSION cuda-cublas-dev-$CUDA_PKG_VERSION cuda-curand-dev-$CUDA_PKG_VERSION
  sudo ln -s /usr/local/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} /usr/local/cuda
fi

# Eigen
mkdir eigen
cd eigen
wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
unzip eigen-b2e267dc99d4.zip
mkdir build && cd build
cmake ..
sudo make install

