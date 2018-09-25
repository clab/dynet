# MNIST Benchmarks

Here is the comparison between Dynet and PyTorch on the "Hello World" example of deep learning : MNIST digit classification.

## Usage (Dynet)

Download the MNIST dataset from the [official website](http://yann.lecun.com/exdb/mnist/) and decompress it.

<pre>
wget -O - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images.idx3-ubyte
wget -O - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels.idx1-ubyte
wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images.idx3-ubyte
wget -O - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels.idx1-ubyte
</pre>

Install the GPU version of Dynet according to the instructions on the [official website](http://dynet.readthedocs.io/en/latest/python.html#installing-a-cutting-edge-and-or-gpu-version).

The architecture of the Convolutional Neural Network follows the architecture used in the [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/layers).

Here are two Python scripts for Dynet. One (`mnist_dynet_minibatch.py`) applies minibatch, and the other one (`mnist_dynet_autobatch.py`) applies autobatch.

Then, run the training:
<pre>
python mnist_dynet_minibatch.py --dynet_gpus 1
</pre>
or
<pre>
python mnist_dynet_autobatch.py --dynet_gpus 1 --dynet_autobatch 1
</pre>

## Usage (PyTorch)

The code of `mnist_pytorch.py` follows the same line as that of `main.py` in [PyTorch Examples](https://github.com/pytorch/examples/tree/master/mnist). We changed the network architecture as follows in order to match the architecture used in the [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/layers).

<pre>
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10, bias=False)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.4)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
</pre>

Install CUDA version of PyTorch according to the instructions on the [official website](http://pytorch.org/).

Then, run the training:

<pre>
python mnist_pytorch.py
</pre>

## Benchmark

Batch size: 64, learning rate: 0.01. 

| OS | Device | Framework | Speed | Accuracy (After 20 Epochs)|
| --- | --- | --- | --- | --- |
| Ubuntu 16.04 |  GeForce GTX 1080 Ti | PyTorch | ~ 4.49±0.11 s per epoch | 98.95% |
| Ubuntu 16.04 |  GeForce GTX 1080 Ti | DyNet (autobatch) | ~ 8.58±0.09 s per epoch | 98.98% |
| Ubuntu 16.04 |  GeForce GTX 1080 Ti | DyNet (minibatch) | ~ 4.13±0.13 s per epoch | 98.99% |
