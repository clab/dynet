#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ -n "$TRAVIS_TAG" ]]; then
  sed -i.bak "s/# version=.*/version=\"$TRAVIS_TAG\",/" setup.py
  sed -i.bak "s/ -march=native//" CMakeLists.txt
fi

