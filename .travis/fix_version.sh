#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ -n "$TRAVIS_TAG" ]]; then
  VERSION="$TRAVIS_TAG"
else
  VERSION="0.0.0+git.$(git describe --tags --always)"
fi
sed -i.bak "s/# version=.*/version=\"$VERSION\",/" setup.py
sed -i.bak "s/ -march=native//" CMakeLists.txt

