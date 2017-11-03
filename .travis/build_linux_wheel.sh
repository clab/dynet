#!/bin/bash
set -xe

# To be run inside docker container
export CMAKE=cmake28 EIGEN3_INCLUDE_DIR="$TRAVIS_BUILD_DIR/eigen" LD_LIBRARY_PATH="$TRAVIS_BUILD_DIR/build/dynet"
cd "$TRAVIS_BUILD_DIR"

# Compile wheels
for PYBIN in /opt/python/*${PYVER/./}*/bin; do
  "$PYBIN/pip" install -U numpy pypandoc twine cython
  if [[ "$BUILD_ARCH" != i686 ]]; then
    yum install -y gmp-devel
    "$PYBIN/python" -c 'from pypandoc.pandoc_download import *; download_pandoc()'
  fi || true  # It's ok if we fail installing pandoc; only important for deployment
  if [[ -n "$DYNET_TEST" ]]; then
    "$TRAVIS_BUILD_DIR"/.travis/test_dynet.sh
  else  # build
    "$PYBIN/python" setup.py bdist_wheel
  fi
done

# Bundle external shared libraries into the wheels
for whl in build/py*/python/dist/*.whl; do
  auditwheel repair "$whl"
done
mv wheelhouse dist

# Fix permissions for any new files because user is root in Docker
cd "$TRAVIS_BUILD_DIR"
chmod -R a+rw .
find . -type d -exec chmod a+x {} \;
