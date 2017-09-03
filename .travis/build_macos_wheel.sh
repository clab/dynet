#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel || travis_terminate 1
delocate-wheel -v dist/*.whl

