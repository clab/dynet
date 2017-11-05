#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel
mkdir -p dist
for whl in build/py*/python/dist/*.whl; do
  delocate-wheel -v "$whl" -w dist
done

