#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel
DYLIB="$TRAVIS_BUILD_DIR/miniconda/envs/$PYVER/lib/libdynet.dylib"
mkdir -p dist
for whl in build/py*/python/dist/*.whl; do
  echo Trying to relink $DYLIB into $whl...
  DEPS_BEFORE="$(delocate-listdeps $whl)"
  delocate-wheel -v "$whl" -w dist
  DEPS_AFTER="$(delocate-listdeps $whl)"
  if [[ "$DEPS_BEFORE" == "$DEPS_AFTER" ]]; then
    echo "Failed fixing wheel, see https://github.com/clab/dynet/pull/841"
    mkdir -p dist
    cp -vf "$whl" dist/
  else
    echo "Success!"
  fi
done

