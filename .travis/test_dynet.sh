#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ "$BUILD_ARCH" == i686 && `arch` == x86_64 ]]; then
  echo Skipping Python test on $BUILD_ARCH
  #echo Using Docker to test $BUILD_ARCH build on `arch`
  #docker run -e PYVER -e TRAVIS_OS_NAME -e CONDA_PACKAGES -e TRAVIS_BUILD_DIR=/build -e DYNET_TEST=1 -v "$TRAVIS_BUILD_DIR":/build --rm "dynet-manylinux1-${BUILD_ARCH}-builder" /root/.travis/build_manylinux_wheel.sh
else
  if [[ "$PYTHON_INSTALL" == manual ]]; then
    cd build
    if [[ "$TRAVIS_OS_NAME" == linux ]]; then
      make -j$(nproc)
    elif [[ "$TRAVIS_OS_NAME" == osx ]]; then
      source activate "$PYVER"
      make -j$(sysctl -n hw.ncpu)
      export DYLD_LIBRARY_PATH="$TRAVIS_BUILD_DIR/build/dynet"
    fi
    if [[ "$BACKEND" != cuda ]]; then
      make install
      export CTEST_OUTPUT_ON_FAILURE=1
      make test
      cd python
      python ../../setup.py build --build-dir=.. --skip-build install --user
    fi
  else  # PYTHON_INSTALL is pip
    if [[ "$TRAVIS_OS_NAME" == osx ]]; then
      source activate "$PYVER"
    fi
    pip install dynet --no-index -f dist
  fi

  if [[ "$BACKEND" != cuda ]]; then
    cd "$TRAVIS_BUILD_DIR"/tests/python
    python test.py
  fi
fi
