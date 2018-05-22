# Translation with a Sequence to Sequence Network and Attention

Here is a Dynet version of the [PyTorch tutorial example "Translation with a Sequence to Sequence Network and Attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

## Data Preparation

Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory.

We use a helper class called `Lang` which has `word2index` and `index2word` dictionaries, as well as a count of each word `word2count` to use to later replace rare words.
