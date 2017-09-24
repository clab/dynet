#!/bin/bash
set -xe

# To be run inside docker container

"$TRAVIS_BUILD_DIR"/.travis/install_conda.sh
export PATH="$PWD/miniconda/bin:$PATH" CMAKE=cmake28 EIGEN3_INCLUDE_DIR="$TRAVIS_BUILD_DIR/eigen"
cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
if [[ -n "$DYNET_TEST" ]]; then
  "$TRAVIS_BUILD_DIR"/.travis/test_dynet.sh
else  # build
  pip install pypandoc  # needed to generate the description from the readme
  python setup.py bdist_wheel
  source deactivate  # auditwheel installed in system but not in conda env
  export LD_LIBRARY_PATH="$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER/lib" 
  cd ./build/py*/python
  auditwheel repair dist/*.whl
  rm -f dist/*.whl
  mv -f wheelhouse/*.whl dist/
  chmod -R a+rw . "$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER"
  find . "$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER" -type d -exec chmod a+x {} \;
fi

