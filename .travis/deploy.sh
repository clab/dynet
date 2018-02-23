#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ -n "$TRAVIS_TAG" ]]; then
  REPO=  # Upload to PyPI
else
  REPO="--repository-url https://test.pypi.org/legacy/"  # Upload to TestPyPI
fi
twine upload $REPO --skip-existing dist/*

