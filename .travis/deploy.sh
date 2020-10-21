#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
twine check dist/*
twine upload --verbose --skip-existing dist/*  # Upload to PyPI
