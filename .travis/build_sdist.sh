#!/bin/bash
set -xe

conda create -q -n sdist
source activate sdist
pip install pypandoc
cd "$TRAVIS_BUILD_DIR"
python setup.py sdist

