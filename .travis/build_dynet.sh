#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ "$BACKEND" == cuda ]]; then
  sed -i.bak "s/\(APPEND CUDA_NVCC_FLAGS.*;\)/\1-w;/" dynet/CMakeLists.txt
  export DYNET_TEST_DEVICES=CPU
  BACKEND_OPTION="-DBACKEND=cuda"
fi
if [[ "$PYTHON_INSTALL" == manual ]]; then
  mkdir build
  cd build
  cmake .. $BACKEND_OPTION -DEIGEN3_INCLUDE_DIR="$EIGEN3_INCLUDE_DIR" -DENABLE_BOOST=ON -DENABLE_CPP_EXAMPLES=ON -DENABLE_C=ON -DPYTHON=$(which python) -DCMAKE_INSTALL_PREFIX=$(dirname $(which python))/..
else  # pip
  .travis/fix_version.sh
  if [[ "$TRAVIS_OS_NAME" == linux ]]; then
    docker build --rm -t "dynet-manylinux1-${BUILD_ARCH}-builder" -f "docker/Dockerfile-$BUILD_ARCH" .
    docker run -e PYVER -e BUILD_ARCH -e TRAVIS_BUILD_DIR=/build -v "$TRAVIS_BUILD_DIR":/build --rm "dynet-manylinux1-${BUILD_ARCH}-builder" /root/.travis/build_linux_wheel.sh
  elif [[ "$TRAVIS_OS_NAME" == osx ]]; then
    .travis/build_macos_wheel.sh
  fi
fi

