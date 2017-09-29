#!/bin/bash
set -xe

if [[ "$BUILD_ARCH" == i686 && `arch` == x86_64 ]]; then
  echo Skipping Python test on $BUILD_ARCH
  #echo Using Docker to test $BUILD_ARCH build on `arch`
  #docker run -e PYVER -e TRAVIS_OS_NAME -e CONDA_PACKAGES -e TRAVIS_BUILD_DIR=/build -e DYNET_TEST=1 -v "$TRAVIS_BUILD_DIR":/build --rm "dynet-manylinux1-${BUILD_ARCH}-builder" /root/.travis/build_manylinux_wheel.sh
else
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
    if [[ "$BACKEND" != cuda ]]; then
      make install
      export CTEST_OUTPUT_ON_FAILURE=1
      make test
      cd python
      python ../../setup.py build --build-dir=.. --skip-build install --user
    fi
  else
    pip install dynet --no-index -f dist
    if [[ "$TRAVIS_OS_NAME" == osx ]]; then
      export DYLD_LIBRARY_PATH=$(dirname $(which python))/../lib:$DYLD_LIBRARY_PATH
    fi
  fi

  if [[ "$BACKEND" != cuda ]]; then
    cd "$TRAVIS_BUILD_DIR"/tests/python
    python test.py
  fi
fi
