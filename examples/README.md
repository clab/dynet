# DyNet Examples

This is a set of common (and less common) models and their implementation in Dynet (C++ and Python).

Some examples have only one of the two languages, or lack documentation, in which case we welcome contributions for the other.
Documentation should include directions on how to download standard datasets, run these examples on these datasets, and calculate standard measures of accuracy etc.
A good example of a simple README is in the [mnist](https://github.com/clab/dynet/tree/master/examples/mnist/) directory.
Contributions to adding these are welcome!

Note that these examples are meant to be minimal examples, not necessarily the state of the art.
Concurrently, we are working on creating a state-of-the-art model repository.
In the mean time, you can browse the [many research projects](http://dynet.io/complex/) that use DyNet and find one that fits your needs.

## Main Model Examples

These examples are of common models and are intended to be relatively well maintained.

* [XOR](https://github.com/clab/dynet/tree/master/examples/xor/): The simplest possible model, solving xor (C++/Python).
* [MNIST](https://github.com/clab/dynet/tree/master/examples/mnist/): An example of MNIST image classification using a simple multi-layer perceptron (C++).
* [RNN Language Model](https://github.com/clab/dynet/tree/master/examples/rnnlm/): A recurrent neural network language model (C++/Python).
* [Sequence-to-sequence Model](https://github.com/clab/dynet/tree/master/examples/sequence-to-sequence/): Sequence to sequence models using standard encoder decoders, or attention (C++/Python).
* [BiLSTM Tagger](https://github.com/clab/dynet/tree/master/examples/tagger/): Models that do sequence labeling with BiLSTM feature extractors (C++/Python).
* [Text Categorization](https://github.com/clab/dynet/tree/master/examples/textcat/): Models for text categorization (C++/Python).
* [Word Embedding](https://github.com/clab/dynet/tree/master/examples/word-embedding/): Models for word embedding (C++).

## Functionality Examples

These examples demonstrate how to take advantage of various types of functionality of DyNet.

* [Batching](https://github.com/clab/dynet/tree/master/examples/batching/): How to use mini-batch training (C++/Python).
* [Automatic Batching](https://github.com/clab/dynet/tree/master/examples/autobatch/): How to use DyNet's automatic batching functionality (C++).
* [Devices](https://github.com/clab/dynet/tree/master/examples/devices/): How to use DyNet on CPUs, GPUs, or multiple devices (C++/Python).
* [Multiprocessing](https://github.com/clab/dynet/tree/master/examples/multiprocessing/): DyNet's multiprocessing functionality for training models in parallel (C++).
* [TensorBoard](https://github.com/clab/dynet/tree/master/examples/tensorboard/): How to use DyNet with TensorBoard through PyCrayon (Python).
* [Reading/Writing](https://github.com/clab/dynet/tree/master/examples/read-write/): How to read/write models (C++).
* [Jupyter Tutorials](https://github.com/clab/dynet/tree/master/examples/jupyter-tutorials/): Various tutorials in the form of Jupyter notebooks (Python).

## Auxiliary Model Examples

These are somewhat less common and not necessarily well supported, but still may be useful for some people.

* [Document Classification](https://github.com/clab/dynet/tree/master/examples/document-classification/): An example of modeling documents with a hierarchical model (C++).
* [Feed Forward Language Model](https://github.com/clab/dynet/tree/master/examples/fflm/): A model for predicting the next word using a feed forward network (C++).
* [Poisson Regression](https://github.com/clab/dynet/tree/master/examples/poisson-regression/): A model for predicting an integer using Poisson regression given a sentence (C++).
* [Sentence Embedding](https://github.com/clab/dynet/tree/master/examples/sentence-embedding/): A model for learning sentence embeddings from parallel data, with negative sampling (C++).
* [Variational Auto-encoders](https://github.com/clab/dynet/tree/master/examples/variational-autoencoder/): Examples using variational auto-encoders (C++).
* [Noise Contrastive Estimation](https://github.com/clab/dynet/tree/master/examples/noise-contrastive-estimation/): Examples using noise contrastive estimation to speed training (C++).
* [Softmax Builders](https://github.com/clab/dynet/tree/master/examples/softmax-builders/): Examples of how to use other types of softmax functions, including class factored softmax (C++).
* [Segmental RNNs](https://github.com/clab/dynet/tree/master/examples/segmental-rnn/): A segmental RNN model (C++).

