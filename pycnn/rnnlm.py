from pycnn import *
import time
import random

LAYERS = 2
INPUT_DIM = 256  #256
HIDDEN_DIM = 1024  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import sys
import util

class RNNLanguageModel:
    def __init__(self, model, cg, builder=SimpleRNNBuilder):
        self.cg = cg
        self.m = model
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        model.add_lookup_parameters("lookup", (VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", (VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        cg = self.cg.renew()
        builder = self.builder
        builder.new_graph(cg)  # TODO WHY?
        builder.start_new_sequence()

        R = cg.parameters(self.m["R"])
        bias = cg.parameters(self.m["bias"])
        errs = [] # will hold expressions
        es=[]
        for (cw,nw) in zip(sent,sent[1:]):
            # assume word is already a word-id
            x_t = cg.lookup(self.m["lookup"], int(cw))
            y_t = builder.add_input(x_t) 
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = esum(errs)
        return nerr

    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        cg = self.cg.renew()
        builder = self.builder
        builder.new_graph(cg)
        builder.start_new_sequence()

        R = cg.parameters(self.m["R"])
        bias = cg.parameters(self.m["bias"])
        cw = first
        while True:
            x_t = cg.lookup(self.m["lookup"], cw)
            y_t = builder.add_input(x_t)
            r_t = bias + (R * y_t)
            ydist = softmax(r_t)
            dist = cg.inc_forward_vec()
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

if __name__ == '__main__':
    train = util.CharsCorpusReader(sys.argv[1],begin="<s>")
    vocab = util.Vocab.from_corpus(train)
    
    VOCAB_SIZE = vocab.size()

    model = Model()
    sgd = SimpleSGDTrainer(model)
    cg = ComputationGraph()

    lm = RNNLanguageModel(model, cg, builder=LSTMBuilder)

    train = list(train)

    chars = loss = 0.0
    for ITER in xrange(100):
        random.shuffle(train)
        for i,sent in enumerate(train):
            _start = time.time()
            if i % 50 == 0:
                sgd.status()
                if chars > 0: print loss / chars,
                for _ in xrange(1):
                    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
                    print "".join([vocab.i2w[c] for c in samp]).strip()
                loss = 0.0
                chars = 0.0
                
            chars += len(sent)-1
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.BuildLMGraph(isent)
            loss += cg.inc_forward_scalar()
            cg.backward()
            sgd.update(1.0)
            print "TM:",(time.time() - _start)/len(sent)
        print "ITER",ITER,loss
        sgd.status()
        sgd.update_epoch(1.0)
