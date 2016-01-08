# a version rnnlm.py using the transduce() interface.
from pycnn import *
import time
import random

LAYERS = 2
INPUT_DIM = 50  #256
HIDDEN_DIM = 50  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count, izip
import sys
import util

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.m = model
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        model.add_lookup_parameters("lookup", (VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", (VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        renew_cg()
        init_state = self.builder.initial_state()

        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        errs = [] # will hold expressions
        es=[]
        state = init_state
        lookup = self.m["lookup"]
        inputs = [lookup[int(cw)] for cw in sent[:-1]]
        expected_outputs = [int(nw) for nw in sent[1:]]
        outputs = state.transduce(inputs)
        r_ts = ((bias + (R * y_t)) for y_t in outputs)
        errs = [pickneglogsoftmax(r_t, eo) for r_t, eo in izip(r_ts, expected_outputs)]
        nerr = esum(errs)
        return nerr

    def sample(self, first=1, nchars=0, stop=-1):
        # sampling must use the regular incremental interface.
        res = [first]
        renew_cg()
        state = self.builder.initial_state()

        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        cw = first
        while True:
            x_t = lookup(self.m["lookup"], cw)
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            ydist = softmax(r_t)
            dist = ydist.vec_value()
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

    #lm = RNNLanguageModel(model, builder=LSTMBuilder)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder)

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
            loss += errs.scalar_value()
            errs.backward()
            sgd.update(1.0)
            #print "TM:",(time.time() - _start)/len(sent)
        print "ITER",ITER,loss
        sgd.status()
        sgd.update_epoch(1.0)
