#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel
export LD_LIBRARY_PATH="$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER/lib"
delocate-wheel -v dist/*.whl

