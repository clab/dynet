#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel
delocate-wheel -v dist/*.whl

