#!/bin/bash
set -xe

hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb
cd eigen
mkdir build && cd build
cmake ..
sudo make install

