# MNIST Benchmarks

Here is the comparison between Dynet and Pytorch on the "Hello World" example of deep learning : MNIST digit classification.

## Usage (Dynet)

Download the MNIST dataset from the [official website](http://yann.lecun.com/exdb/mnist/) and decompress it.

    wget -O - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images.idx3-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels.idx1-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images.idx3-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels.idx1-ubyte

Install the GPU version of Dynet according to the instructions on the [official website](http://dynet.readthedocs.io/en/latest/python.html#installing-a-cutting-edge-and-or-gpu-version).

The architecture of the Convolutional Neural Network follows the architecture used in the [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/).

Here are two Python scripts for Dynet. One (mnist_dynet_minibatch.py) applies minibatch, and the other one (mnist_dynet_autobatch.py) applies autobatch.


Then, run the training (here for a batch size of 128 and 20 epochs) :

    ./train_mnist \
    --train train-images.idx3-ubyte \
    --train_labels train-labels.idx1-ubyte \
    --dev t10k-images.idx3-ubyte \
    --dev_labels t10k-labels.idx1-ubyte \
    --batch_size 128 \
    --num_epochs 20

## Benchmark

System | Speed | Test accuracy (after 20 epochs)
------------ | ------------- | -------------
Intel® Core™ i5-4200H CPU @ 2.80GHz × 4 | ~7±0.5 s per epoch| 97.84 %

