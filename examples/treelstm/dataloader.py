import re
import codecs
import random
import numpy as np
from collections import Counter


def read_dataset(filename):
    return [Tree.from_sexpr(line.strip()) for line in codecs.open(filename, "r")]


def get_vocabs(trees):
    label_vocab = Counter()
    word_vocab = Counter()
    for tree in trees:
        label_vocab.update([n.label for n in tree.nonterms()])
        word_vocab.update([l.label for l in tree.leaves()])
    words = ["_UNK_"] + [x for x, c in word_vocab.items() if c > 0]
    w2i = {w: i for i, w in enumerate(words)}
    return w2i, words


def get_embeds(vocab, embed_path):
    total, hit = len(vocab), 0
    shape = (total, 300)
    word_embeds = np.random.randn(np.product(shape)).reshape(shape)
    with codecs.open(embed_path) as f:
        for line in f:
            line = line.strip().split(' ')
            word, embed = line[0], line[1:]
            idx = vocab.get(word, -1)
            if idx != -1:
                hit += 1
                word_embeds[idx]= np.array([np.float32(x) for x in embed])
    print('{}/{} embeddings hit in glove'.format(hit, total))
    return word_embeds


def _tokenize_sexpr(s):
    tokker = re.compile(r" +|[()]|[^ ()]+")
    toks = [t for t in [match.group(0) for match in tokker.finditer(s)] if t[0] != " "]
    return toks


def _within_bracket(toks):
    label = next(toks)
    children = []
    for tok in toks:
        if tok == "(":
            children.append(_within_bracket(toks))
        elif tok == ")":
            return Tree(label, children)
        else:
            children.append(Tree(tok, None))
    assert (False), list(toks)


class Tree(object):
    def __init__(self, label, children=None):
        self.label = label if children is None else int(label)
        self.children = children

    @staticmethod
    def from_sexpr(string):
        toks = iter(_tokenize_sexpr(string))
        assert next(toks) == "("
        return _within_bracket(toks)

    def __str__(self):
        if self.children is None: return self.label
        return "[%s %s]" % (self.label, " ".join([str(c) for c in self.children]))

    def isleaf(self):
        return self.children is None

    def leaves_iter(self):
        if self.isleaf():
            yield self
        else:
            for c in self.children:
                for l in c.leaves_iter(): yield l

    def leaves(self):
        return list(self.leaves_iter())

    def nonterms_iter(self):
        if not self.isleaf():
            yield self
            for c in self.children:
                for n in c.nonterms_iter(): yield n

    def nonterms(self):
        return list(self.nonterms_iter())


class DataLoader(object):
    def __init__(self, datapath, statistics=False):
        self.data = read_dataset(datapath)
        self.n_samples = len(self.data)
        self.reset()
        if statistics:
            self.w2i, self.i2w = get_vocabs(self.data)

    def reset(self, shuffle=True):
        self.idx = 0
        if shuffle: random.shuffle(self.data)

    def __iter__(self):
        while self.idx < self.n_samples:
            yield self.data[self.idx]
            self.idx += 1

    def batches(self, batch_size=25):
        while self.idx < self.n_samples:
            yield self.data[self.idx: self.idx + batch_size]
            self.idx += batch_size
