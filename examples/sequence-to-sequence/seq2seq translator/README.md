# Translation with a Sequence to Sequence Network and Attention

Here is a Dynet version of the [PyTorch tutorial example "Translation with a Sequence to Sequence Network and Attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

## Data Preparation

The data for this project is a set of many thousands of English to French translation pairs. Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory.

We use a helper class called `Lang` which has `word2index` and `index2word` dictionaries, as well as a count of each word `word2count` to use to later replace rare words.

<pre>
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
</pre>
