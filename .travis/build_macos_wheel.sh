#!/bin/bash
set -xe

cd "$TRAVIS_BUILD_DIR"
source activate "$PYVER"
pip install pypandoc delocate
python setup.py bdist_wheel
DYLIB="$TRAVIS_BUILD_DIR"/miniconda/envs/"$PYVER"/lib/libdynet.dylib
WHEEL=dist/*.whl
echo Trying to relink $DYLIB into $WHEEL...
DEPS_BEFORE="$(delocate-listdeps $WHEEL)"
delocate-wheel -v "$WHEEL"
DEPS_AFTER="$(delocate-listdeps $WHEEL)"
if [[ "$DEPS_BEFORE" == "$DEPS_AFTER" ]]; then
  echo "Failed fixing wheel, see https://github.com/clab/dynet/pull/841"
else
  echo "Success!"
fi

