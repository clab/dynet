import dynet as dy
import time
import random

LAYERS = 2
INPUT_DIM = 256 #50  #256
HIDDEN_DIM = 256 # 50  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import argparse
import sys
import util

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.SimpleRNNBuilder):
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def save_to_disk(self, filename):
        dy.save(filename, [self.builder, self.lookup, self.R, self.bias])

    def load_from_disk(self, filename):
        (self.builder, self.lookup, self.R, self.bias) = dy.load(filename, model)
        
    def build_lm_graph(self, sent):
        dy.renew_cg()
        init_state = self.builder.initial_state()

        errs = [] # will hold expressions
        es=[]
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            # assume word is already a word-id
            x_t = dy.lookup(self.lookup, int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = self.bias + (self.R * y_t)
            err = dy.pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = dy.esum(errs)
        return nerr
    
    def predict_next_word(self, sentence):
        dy.renew_cg()
        init_state = self.builder.initial_state()
        state = init_state
        for cw in sentence:
            # assume word is already a word-id
            x_t = self.lookup[int(cw)]
            state = state.add_input(x_t)
        y_t = state.output()
        r_t = self.bias + (self.R * y_t)
        prob = dy.softmax(r_t)
        return prob
    
    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        dy.renew_cg()
        state = self.builder.initial_state()

        cw = first
        while True:
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = self.bias + (self.R * y_t)
            ydist = dy.softmax(r_t)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help='Path to the corpus file.')
    args = parser.parse_args()

    train = util.CharsCorpusReader(args.corpus, begin="<s>")
    vocab = util.Vocab.from_corpus(train)
    
    VOCAB_SIZE = vocab.size()

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=1.0)

    #lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.SimpleRNNBuilder)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.LSTMBuilder)

    train = list(train)

    chars = loss = 0.0
    for ITER in range(100):
        random.shuffle(train)
        for i,sent in enumerate(train):
            _start = time.time()
            if i % 50 == 0:
                trainer.status()
                if chars > 0: print(loss / chars,)
                for _ in range(1):
                    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
                    print("".join([vocab.i2w[c] for c in samp]).strip())
                loss = 0.0
                chars = 0.0
                
            chars += len(sent)-1
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.build_lm_graph(isent)
            loss += errs.scalar_value()
            errs.backward()
            trainer.update()
            #print "TM:",(time.time() - _start)/len(sent)
        print("ITER {}, loss={}".format(ITER, loss))
        trainer.status()

    lm.save_to_disk("RNNLanguageModel.model")

    print("loading the saved model...")
    lm.load_from_disk("RNNLanguageModel.model")
    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
    print("".join([vocab.i2w[c] for c in samp]).strip())
 
