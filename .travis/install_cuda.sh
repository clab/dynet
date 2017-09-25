#!/bin/bash
set -xe

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

