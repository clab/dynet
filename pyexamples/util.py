from collections import defaultdict
from itertools import count
class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

class CorpusReader:
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in file(self.fname):
            line = line.strip().split()
            #line = [' ' if x == '' else x for x in line]
            yield line

class CharsCorpusReader:
    def __init__(self, fname, begin=None):
        self.fname = fname
        self.begin = begin
    def __iter__(self):
        begin = self.begin
        for line in file(self.fname):
            line = list(line)
            if begin:
                line = [begin] + line
            yield line
