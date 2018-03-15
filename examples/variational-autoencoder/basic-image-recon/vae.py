from __future__ import print_function
from utils import load_mnist, make_grid, pre_pillow_float_img_process, save_image

import numpy as np
import argparse
import dynet as dy

import os
if not os.path.exists('results'):
    os.makedirs('results')

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dynet-gpu', action='store_true', default=False,
                    help='enables DyNet CUDA training')
parser.add_argument('--dynet-gpus', type=int, default=1, metavar='N',
                    help='number of gpu devices to use')
parser.add_argument('--dynet-seed', type=int, default=None, metavar='N',
                    help='random seed (default: random inside DyNet)')
parser.add_argument('--dynet-mem', type=int, default=None, metavar='N',
                    help='allocating memory (default: default of DyNet 512MB)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

train_data = load_mnist('training', './data')
batch_size = args.batch_size

test_data = load_mnist('testing', './data')


def generate_batch_loader(data, batch_size):
    i = 0
    n = len(data)

    while i + batch_size <= n:
        yield np.asarray(data[i:i+batch_size])
        i += batch_size

    # if i < n:
    #     pass  # last short batch ignored
    #     # yield data[i:]


class DynetLinear:

    def __init__(self, dim_in, dim_out, dyParameterCollection):

        assert(isinstance(dyParameterCollection, dy.ParameterCollection))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.pW = dyParameterCollection.add_parameters((dim_out, dim_in))
        self.pb = dyParameterCollection.add_parameters((dim_out))

    def __call__(self, x):

        assert(isinstance(x, dy.Expression))

        self.W = dy.parameter(self.pW)  # add parameters to graph as expressions # m2.add_parameters((8, len(inputs)))
        self.b = dy.parameter(self.pb)
        self.x = x

        return self.W * self.x + self.b


pc = dy.ParameterCollection()


class VAE:
    def __init__(self, dyParameterCollection):

        assert (isinstance(dyParameterCollection, dy.ParameterCollection))

        self.fc1 = DynetLinear(784, 400, dyParameterCollection)
        self.fc21 = DynetLinear(400, 20, dyParameterCollection)
        self.fc22 = DynetLinear(400, 20, dyParameterCollection)
        self.fc3 = DynetLinear(20, 400, dyParameterCollection)
        self.fc4 = DynetLinear(400, 784, dyParameterCollection)

        self.relu = dy.rectify
        self.sigmoid = dy.logistic

        self.training = False

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = dy.exp(logvar * 0.5)
            eps = dy.random_normal(dim=std.dim()[0], mean=0.0, stddev=1.0)
            return dy.cmult(eps, std) + mu
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        assert(isinstance(x, dy.Expression))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(pc)
optimizer = dy.AdamTrainer(pc, alpha=1e-3)  # alpha: initial learning rate


# # Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = dy.binary_log_loss(recon_x, x)  # equiv to torch.nn.functional.binary_cross_entropy(?,?, size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * dy.sum_elems(1 + logvar - dy.pow(mu, dy.scalarInput(2)) - dy.exp(logvar))

    return BCE + KLD


def train(epoch):
    model.training = True
    train_loss = 0
    train_loader = generate_batch_loader(train_data, batch_size=batch_size)
    for batch_idx, data in enumerate(train_loader):

        # Dymanic Construction of Graph
        dy.renew_cg()
        x = dy.inputTensor(data.reshape(-1, 784).T)
        recon_x, mu, logvar = model.forward(x)
        loss = loss_function(recon_x, x, mu, logvar)

        # Forward
        loss_value = loss.value()
        train_loss += loss_value
        # Backward
        loss.backward()
        optimizer.update()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data),
                100. * batch_idx / (len(train_data) / batch_size),
                loss_value / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))


def test(epoch):
    model.training = False
    test_loss = 0
    test_loader = generate_batch_loader(test_data, batch_size=batch_size)
    for i, data in enumerate(test_loader):

        # Dymanic Construction of Graph
        dy.renew_cg()
        x = dy.inputTensor(data.reshape(-1, 784).T)
        recon_x, mu, logvar = model.forward(x)
        loss = loss_function(recon_x, x, mu, logvar)

        # Forward
        loss_value = loss.value()
        test_loss += loss_value

        if i == 0:
            n = min(data.shape[0], 8)
            comparison = np.concatenate([data[:n],
                                         recon_x.npvalue().T.reshape(batch_size, 1, 28, 28)[:n]])
            save_image(comparison,
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_data)
    print('====> Test set loss: {:.4f}'.format(test_loss))


import time
tictocs = []

for epoch in range(1, args.epochs + 1):
    tic = time.time()

    train(epoch)
    test(epoch)
    sample = dy.inputTensor(np.random.randn(20, 64))
    sample = model.decode(sample)
    save_image(sample.npvalue().T.reshape(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')

    toc = time.time()
    tictocs.append(toc - tic)

print('############\n\n')
print('Total Time Cost:', np.sum(tictocs))
print('Epoch Time Cost', np.average(tictocs), '+-', np.std(tictocs) / np.sqrt(len(tictocs)))
print('\n\n############')
