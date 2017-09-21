#!/bin/bash
set -xe

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
  sudo apt-get -qq update
  sudo apt-get install -y --allow-unauthenticated gcc-4.8 g++-4.8 libboost-filesystem1.55-dev libboost-program-options1.55-dev libboost-serialization1.55-dev libboost-test1.55-dev libboost-regex1.55-dev pandoc
else
  brew install pandoc
fi
