#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
python setup.py sdist

