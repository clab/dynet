#!/bin/bash
set -xe

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  export MINICONDA_OS_NAME=Linux
  if [[ `arch` == x86_64 ]]; then
    export MINICONDA_ARCH=x86_64
  else
    export MINICONDA_ARCH=x86
  fi
else
  export MINICONDA_OS_NAME=MacOSX MINICONDA_ARCH=x86_64
fi
wget "https://repo.continuum.io/miniconda/Miniconda3-latest-$MINICONDA_OS_NAME-$MINICONDA_ARCH.sh" -O miniconda.sh
bash miniconda.sh -b -p miniconda
export PATH="$PWD/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda create -q -n "$PYVER" python="$PYVER" $CONDA_PACKAGES
# Useful for debugging any issues with conda
conda info -a
