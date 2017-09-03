#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"

if [[ "$PYTHON_INSTALL" == manual ]]; then
  cd build
  if [[ "$TRAVIS_OS_NAME" == linux ]]; then
    make -j$(nproc)
  elif [[ "$TRAVIS_OS_NAME" == osx ]]; then
    make -j$(sysctl -n hw.ncpu)
    export DYLD_LIBRARY_PATH=$TRAVIS_BUILD_DIR/build/dynet
  fi
  make install
  make test
  cd python
  python ../../setup.py build --build-dir=.. --skip-build install --user
else
  pip install dynet --no-index -f dist
  if [[ "$TRAVIS_OS_NAME" == osx ]]; then
    export DYLD_LIBRARY_PATH=$(dirname $(which python))/../lib:$DYLD_LIBRARY_PATH
  fi
fi

cd "$TRAVIS_BUILD_DIR"/tests/python
python test.py
