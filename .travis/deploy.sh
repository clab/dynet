#!/bin/bash
set -xe

# Upload to PyPI
conda create -q -n twine
source activate twine
pip install twine
cd "$TRAVIS_BUILD_DIR"
twine upload --skip-existing dist/*
