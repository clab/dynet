#! /usr/bin/env python3

import time
import random
import os
import struct
import argparse
import numpy as np
import dynet as dy

# To run this, download the four files from  http://yann.lecun.com/exdb/mnist/
# using the --download option.
# Pass the path where the data should be stored (or is already stored)
# to the program with the --path option. You will also want to run
# with --dynet_autobatch=1.  To turn on GPU training, run with
# --dynet_gpus=1.

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=".",
                    help="Path to the MNIST data files (unzipped).")
parser.add_argument("--minibatch_size", default=16,
                    help="Size of minibatches.")
parser.add_argument("--conv", dest="conv", action="store_true")
parser.add_argument("--download", dest="download", action="store_true",
                    help="download and extract examples to path")
parser.add_argument("--dynet_autobatch", default=0,
                    help="Set to 1 to turn on autobatching.")
parser.add_argument("--dynet_gpus", default=0,
                    help="Set to 1 to train on GPU.")

HIDDEN_DIM = 1024
DROPOUT_RATE = 0.4

# minimally adapted from https://gist.github.com/akesling/5358964
def read_mnist(dataset, path):
    if dataset is "training":
        fname_img = os.path.join(path, "train-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset is "testing":
        fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.multiply(
            np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows*cols),
            1.0 / 255.0)

    get_instance = lambda idx: (labels[idx], images[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(labels)):
        yield get_instance(i)


class MNISTClassify(object):
  def __init__(self, m):
    if args.conv:
      # architecture from https://www.tensorflow.org/get_started/mnist/pros
      self.F1 = m.add_parameters((5, 5, 1, 32))
      self.b1 = m.add_parameters((32, ))
      self.F2 = m.add_parameters((5, 5, 32, 64))
      self.b2 = m.add_parameters((64, ))
      input_size = 7 * 7 * 64
    else:
      input_size = 28 * 28
    self.W1 = m.add_parameters((HIDDEN_DIM, input_size))
    self.hbias = m.add_parameters((HIDDEN_DIM, ))
    self.W2 = m.add_parameters((10, HIDDEN_DIM))

  def __call__(self, x, dropout=False):
    if args.conv:
      x = dy.reshape(x, (28, 28, 1))
      x = dy.conv2d_bias(x, self.F1, self.b1, [1, 1], is_valid=False)
      x = dy.rectify(dy.maxpooling2d(x, [2, 2], [2, 2]))
      x = dy.conv2d_bias(x, self.F2, self.b2, [1, 1], is_valid=False)
      x = dy.rectify(dy.maxpooling2d(x, [2, 2], [2, 2]))  # 7x7x64
      x = dy.reshape(x, (7 * 7 * 64,))
    h = dy.rectify(self.W1 * x + self.hbias)
    if dropout:
      h = dy.dropout(h, DROPOUT_RATE)
    logits = self.W2 * h
    return logits


def download_examples(path):
    import gzip
    import urllib.request
    baseurl = "http://yann.lecun.com/exdb/mnist/"
    for elem in ["train-images-idx3-ubyte.gz",
                 "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz"]:
        print("downloading " + elem + " ...")
        outfile = open(os.path.join(path, elem[:-3]), "wb")
        downloaded = urllib.request.urlopen(baseurl + elem)
        outfile.write(gzip.GzipFile(fileobj = downloaded).read())
    
if __name__ == "__main__":
  args = parser.parse_args()
  if args.download:
      download_examples(args.path)
  training = [(lbl, img) for (lbl, img) in read_mnist("training", args.path)]
  testing = [(lbl, img) for (lbl, img) in read_mnist("testing", args.path)]

  m = dy.Model()
  classify = MNISTClassify(m)
  sgd = dy.SimpleSGDTrainer(m, learning_rate=0.01)

  eloss = None
  alpha = 0.05  # smoothing of training loss for reporting
  start = time.time()
  dev_time = 0
  report = args.minibatch_size * 30
  dev_report = args.minibatch_size * 600
  for epoch in range(50):
    random.shuffle(training)
    print(("Epoch {} starting".format(epoch+1)))
    i = 0
    while i < len(training):
      dy.renew_cg()
      mbsize = min(args.minibatch_size, len(training) - i)
      minibatch = training[i:i+mbsize]
      losses = []
      for lbl, img in minibatch:
        x = dy.inputVector(img)
        logits = classify(x, dropout=True)
        loss = dy.pickneglogsoftmax(logits, lbl)
        losses.append(loss)
      mbloss = dy.esum(losses) / mbsize
      mbloss.backward()
      sgd.update()

      # eloss is an exponentially smoothed loss.
      if eloss is None:
        eloss = mbloss.scalar_value()
      else:
        eloss = mbloss.scalar_value() * alpha + eloss * (1.0 - alpha)

      # Do dev evaluation here:
      if (i > 0) and (i % dev_report == 0):
        confusion = [[0 for _ in range(10)] for _ in range(10)]
        correct = 0
        dev_start = time.time()
        for s in range(0, len(testing), args.minibatch_size):
          dy.renew_cg()
          e = min(len(testing), s + args.minibatch_size)
          minibatch = testing[s:e]
          scores = []
          for lbl, img in minibatch:
            x = dy.inputVector(img)
            logits = classify(x)
            scores.append((lbl, logits))

          # This evaluates all the logits in a batch if autobatching is on.
          dy.forward([logits for _, logits in scores])

          # now we can retrieve the batch-computed logits cheaply
          for lbl, logits in scores:
            prediction = np.argmax(logits.npvalue())
            if lbl == prediction:
              correct += 1
            confusion[prediction][lbl] += 1
        dev_end = time.time()
        acc = float(correct) / len(testing)
        dev_time += dev_end - dev_start
        print(("Held out accuracy {} ({} instances/sec)".format(
            acc, len(testing) / (dev_end - dev_start))))
        print('   ' + ''.join(('T'+str(x)).ljust(6) for x in range(10)))
        for p, row in enumerate(confusion):
          s = 'P' + str(p) + ' '
          s += ''.join(str(col).ljust(6) for col in row)
          print(s)

      if (i > 0) and (i % report == 0):
        print(("moving avg loss: {}".format(eloss)))
      i += mbsize
    end = time.time()
    print(("instances per sec: {}".format(
        (i + epoch * len(training)) / (end - start - dev_time))))

