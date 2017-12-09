#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
if [[ "$TRAVIS_OS_NAME" == linux ]]; then
  python setup.py sdist  # Build sdist
fi
twine upload --skip-existing dist/*  # Upload to PyPI
