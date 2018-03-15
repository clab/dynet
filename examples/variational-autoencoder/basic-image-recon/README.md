# Basic VAE Example: MNIST Images

This example is a DyNet version of the same example as [a PyTorch one](https://github.com/pytorch/examples/tree/master/vae).
Benchmarks are attached below.

> This is an improved implementation of the paper Stochastic Gradient VB and the Variational Auto-Encoder by Kingma and Welling. It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

### 1. Prepare Data

```sh
mkdir data
wget -O - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > data/train-images-idx3-ubyte
wget -O - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > data/train-labels-idx1-ubyte
wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > data/t10k-images-idx3-ubyte
wget -O - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > data/t10k-labels-idx1-ubyte
```

### 2. Run

```sh
pip install -r requirements.txt
python vae.py

# To support gpu, please refer to the DyNet document,
#   install a gpu version of DyNet with pip,
#   and run >>> python vae.py --dynet-gpu
```

### Benchmarks
| OS | Device | Framework | Speed | Test Loss (After 10 Epochs)|
| --- | --- | --- | --- | --- |
| MAC OS 10.13.3 | 2.3 GHz Intel Core i5 | PyTorch | ~ 12.5±0.3 s per epoch | 97.0215 |
| MAC OS 10.13.3 | 2.3 GHz Intel Core i5 | DyNet | ~ 19.8±0.3 s per epoch | 97.4233 |
| Ubuntu 16.04 |  3.40 GHz Intel Core i7-6800K | PyTorch | ~ 13.38±0.05 s per epoch | 97.1197 |
| Ubuntu 16.04 |  3.40 GHz Intel Core i7-6800K | DyNet | ~ 8.73±0.01 s per epoch | 97.3211 |
| Ubuntu 16.04 |  GeForce GTX 1080 Ti | PyTorch | ~ 4.23±0.01 s per epoch | 97.5330 |
| Ubuntu 16.04 |  GeForce GTX 1080 Ti | DyNet | ~ 2.91±0.01 s per epoch | 97.6848 |
