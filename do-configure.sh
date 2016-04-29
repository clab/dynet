#!/bin/bash

# BOOST_ROOT=/home/is/neubig/usr/local/boost_1_55_0
# cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/usr/local/eigen -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=/home/is/neubig/usr/local/cuda/toolkit7.0/toolkit -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_ROOT:PATHNAME=/home/is/neubig/usr/local/boost_1_55_0 -DBoost_LIBRARY_DIRS:FILEPATH=/home/is/neubig/usr/local/boost_1_55_0/lib
$HOME/usr/bin/cmake .. -DEIGEN3_INCLUDE_DIR=$HOME/usr/local/eigen  -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_ROOT:PATHNAME=/home/is/neubig/usr/local/boost_1_58_0 -DBoost_LIBRARY_DIRS:FILEPATH=/home/is/neubig/usr/local/boost_1_58_0/lib -DBACKEND=cuda
make -j 24
