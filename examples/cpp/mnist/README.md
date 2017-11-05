# MNIST example

Here's an example usage of dynet for the "Hello World" of deep learning : MNIST digit classification

## Usage

First, download the MNIST dataset from the [official website](http://yann.lecun.com/exdb/mnist/) and decompress it.

    wget -O - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images.idx3-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels.idx1-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images.idx3-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels.idx1-ubyte

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
