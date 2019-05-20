#!/bin/bash -x
rm -rf data
mkdir -p data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P data
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P data
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P data
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P data
gunzip data/*
