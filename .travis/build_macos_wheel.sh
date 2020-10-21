#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
python setup.py bdist_wheel
cp -vr build/py*/python/dist ./
