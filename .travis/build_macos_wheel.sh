#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel
cp "$TRAVIS_BUILD_DIR"/miniconda/envs/$PYVER/lib/libdynet.dylib ./
delocate-listdeps --depending dist/*.whl
delocate-wheel -v dist/*.whl

