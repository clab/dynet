#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ "$PYTHON_INSTALL" == manual ]]; then
  mkdir build
  cd build
  cmake .. -DEIGEN3_INCLUDE_DIR="$EIGEN3_INCLUDE_DIR" -DENABLE_BOOST=ON -DENABLE_CPP_EXAMPLES=ON -DPYTHON=$(which python) -DCMAKE_INSTALL_PREFIX=$(dirname $(which python))/..
else  # pip
  if [[ -n "$TRAVIS_TAG" ]]; then
    sed -i.bak "s/# version=.*/version=\"$TRAVIS_TAG\",/" setup.py
  fi
  if [[ "$TRAVIS_OS_NAME" == linux ]]; then
    docker build --rm -t "dynet-manylinux1-${BUILD_ARCH}-builder" -f "docker/Dockerfile-$BUILD_ARCH" .
    docker run -e PYVER -e TRAVIS_OS_NAME -e CONDA_PACKAGES -e TRAVIS_BUILD_DIR=/build -v "$TRAVIS_BUILD_DIR":/build --rm "dynet-manylinux1-${BUILD_ARCH}-builder" /root/.travis/build_manylinux_wheel.sh
  elif [[ "$TRAVIS_OS_NAME" == osx ]]; then
    .travis/build_macos_wheel.sh
  fi
fi

