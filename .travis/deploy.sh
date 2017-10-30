#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
python setup.py sdist  # Build sdist
twine upload --skip-existing dist/*  # Upload to PyPI
