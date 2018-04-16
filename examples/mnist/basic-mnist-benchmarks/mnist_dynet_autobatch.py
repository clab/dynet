from __future__ import division
import os
import struct
import argparse
import random
import time
import numpy as np
# import dynet as dy
# import dynet_config
# dynet_config.set_gpu()
import dynet as dy

# First, download the MNIST dataset from the official website and decompress it.
# wget -O - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images.idx3-ubyte
# wget -O - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels.idx1-ubyte
# wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images.idx3-ubyte
# wget -O - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels.idx1-ubyte

parser = argparse.ArgumentParser(description='DyNet MNIST Example')
parser.add_argument("--path", type=str, default=".",
                    help="Path to the MNIST data files (unzipped).")
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument("--dynet_autobatch", type=int, default=0,
                    help="Set to 1 to turn on autobatching.")
parser.add_argument("--dynet_gpus", type=int, default=0,
                    help="Set to 1 to train on GPU.")

HIDDEN_DIM = 1024
DROPOUT_RATE = 0.4

# Adapted from https://gist.github.com/akesling/5358964
def read(dataset, path):
    if dataset is "training":
        fname_img = os.path.join(path, "train-images.idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels.idx1-ubyte")
    elif dataset is "testing":
        fname_img = os.path.join(path, "t10k-images.idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels.idx1-ubyte")
    else:
        raise ValueError("dataset must be 'training' or 'testing'")

    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.multiply(np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols), 1.0/255.0)

    get_img = lambda idx: (lbl[idx], img[idx])

    for i in range(len(lbl)):
        yield get_img(i)
    
class mnist_network(object):
    
    def __init__(self, m):
        self.pConv1 = m.add_parameters((5, 5, 1, 32))
        self.pB1 = m.add_parameters((32, ))
        self.pConv2 = m.add_parameters((5, 5, 32, 64))
        self.pB2 = m.add_parameters((64, ))
        self.pW1 = m.add_parameters((HIDDEN_DIM, 7*7*64))
        self.pB3 = m.add_parameters((HIDDEN_DIM, ))
        self.pW2 = m.add_parameters((10, HIDDEN_DIM))
        
    def __call__(self, inputs, dropout=False):
        x = dy.inputTensor(inputs)
        conv1 = dy.parameter(self.pConv1)
        b1 = dy.parameter(self.pB1)
        x = dy.conv2d_bias(x, conv1, b1, [1, 1], is_valid=False)
        x = dy.rectify(dy.maxpooling2d(x, [2, 2], [2, 2]))
        conv2 = dy.parameter(self.pConv2)
        b2 = dy.parameter(self.pB2)
        x = dy.conv2d_bias(x, conv2, b2, [1, 1], is_valid=False)
        x = dy.rectify(dy.maxpooling2d(x, [2, 2], [2, 2]))
        x = dy.reshape(x, (7*7*64, 1))
        w1 = dy.parameter(self.pW1)
        b3 = dy.parameter(self.pB3)
        h = dy.rectify(w1*x+b3)
        if dropout:
            h = dy.dropout(h, DROPOUT_RATE)
        w2 = dy.parameter(self.pW2)
        output = w2*h
        # output = dy.softmax(w2*h)
        return output
    
    def create_network_return_loss(self, inputs, expected_output, dropout=False):
        out = self(inputs, dropout)
        loss = dy.pickneglogsoftmax(out, expected_output)
        # loss = -dy.log(dy.pick(out, expected_output))
        return loss
        
    def create_network_return_best(self, inputs, dropout=False):
        out = self(inputs, dropout)
        out = dy.softmax(out)
        return np.argmax(out.npvalue())
        # return np.argmax(out.npvalue())
    
args = parser.parse_args()
train_data = [(lbl, img) for (lbl, img) in read("training", args.path)]
test_data = [(lbl, img) for (lbl, img) in read("testing", args.path)]
    
m = dy.ParameterCollection()
network = mnist_network(m)
trainer = dy.SimpleSGDTrainer(m, learning_rate=args.lr)

def train(epoch):
    random.shuffle(train_data)
    i = 0
    epoch_start = time.time()
    while i < len(train_data):
        dy.renew_cg()
        losses = []
        for lbl, img in train_data[i:i+args.batch_size]:
            loss = network.create_network_return_loss(img, lbl, dropout=True)
            losses.append(loss)
        mbloss = dy.average(losses)
        if (int(i/args.batch_size)) % args.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, i, len(train_data),
                                100. * i/len(train_data), mbloss.value()))
        mbloss.backward()
        trainer.update()
        i += args.batch_size
    epoch_end = time.time()
    print("{} s per epoch".format(epoch_end-epoch_start))
        
def test():
    correct = 0
    dy.renew_cg()
    losses = []
    for lbl, img in test_data:
        losses.append(network.create_network_return_loss(img, lbl, dropout=False))
        if lbl == network.create_network_return_best(img, dropout=False):
            correct += 1
    mbloss = dy.average(losses)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            mbloss.value(), correct, len(test_data),
            100. * correct / len(test_data)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    
# m.save("/tmp/tmp.model")
