#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
twine upload --verbose --skip-existing dist/*  # Upload to PyPI
