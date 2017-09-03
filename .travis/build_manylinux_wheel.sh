#!/bin/bash
set -xe

# To be run inside docker container

"$TRAVIS_BUILD_DIR"/.travis/install_conda.sh
export PATH="$PWD/miniconda/bin:$PATH" CMAKE=cmake28 EIGEN3_INCLUDE_DIR="$TRAVIS_BUILD_DIR/eigen"
cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc
python setup.py bdist_wheel
source deactivate
export LD_LIBRARY_PATH="$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER/lib" 
auditwheel repair dist/*.whl
rm -f dist/*.whl
mv -f wheelhouse/*.whl dist/
chmod -R a+rw . "$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER"
find . "$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER" -type d -exec chmod a+x {} \;

