from dynet import *
import time
import random

LAYERS = 2
INPUT_DIM = 256 #50  #256
HIDDEN_DIM = 256 # 50  #1024
VOCAB_SIZE = 0
MB_SIZE = 50  # mini batch size

from collections import defaultdict
from itertools import count
import sys
import util

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder):
        # Char-level LSTM (layers=2, input=256, hidden=128, model)
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
        # Lookup parameters for word embeddings
        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        # Softmax weights/biases on top of LSTM outputs
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    # Build the language model graph
    def BuildLMGraph(self, sents):
        renew_cg()
        # initialize the RNN
        init_state = self.builder.initial_state()
        # parameters -> expressions
        R = parameter(self.R)
        bias = parameter(self.bias)

        S = vocab.w2i["<s>"]
        # get the cids and masks for each step
        tot_chars = 0
        cids = []
        masks = []

        for i in range(len(sents[0])):
            cids.append([(vocab.w2i[sent[i]] if len(sent) > i else S) for sent in sents])
            mask = [(1 if len(sent)>i else 0) for sent in sents]
            masks.append(mask)
            tot_chars += sum(mask)

        # start the rnn with "<s>"
        init_ids = cids[0]
        s = init_state.add_input(lookup_batch(self.lookup, init_ids))

        losses = []

        # feed char vectors into the RNN and predict the next char
        for cid, mask in zip(cids[1:], masks[1:]):
            score = affine_transform([bias, R, s.output()])
            loss = pickneglogsoftmax_batch(score, cid)
            # mask the loss if at least one sentence is shorter
            if mask[-1] != 1:
                mask_expr = inputVector(mask)
                mask_expr = reshape(mask_expr, (1,), len(sents))
                loss = loss * mask_expr

            losses.append(loss)
            # update the state of the RNN
            cemb = lookup_batch(self.lookup, cid)
            s = s.add_input(cemb)

        return sum_batches(esum(losses)), tot_chars


    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        renew_cg()
        state = self.builder.initial_state()

        R = parameter(self.R)
        bias = parameter(self.bias)
        cw = first
        while True:
            x_t = lookup(self.lookup, cw)
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

    #lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=LSTMBuilder)

    train = list(train)
    # Sort training sentences in descending order and count minibatches
    train.sort(key=lambda x: -len(x))
    train_order = [x*MB_SIZE for x in range(int((len(train)-1)/MB_SIZE + 1))]

    # Perform training
    i = 0
    chars = loss = 0.0
    for ITER in range(100):
        random.shuffle(train_order)
        #_start = time.time()
        for sid in train_order: 
            i += 1
            #if i % int(50) == 0:
            sgd.status()
            if chars > 0: print(loss / chars,)
            for _ in range(1):
                samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
                print("".join([vocab.i2w[c] for c in samp]).strip())
            loss = 0.0
            chars = 0.0

            # train on the minibatch
            errs, mb_chars = lm.BuildLMGraph(train[sid: sid + MB_SIZE])
            loss += errs.scalar_value()
            chars += mb_chars
            errs.backward()
            sgd.update()
                

            #print "TM:",(time.time() - _start)/chars
        print("ITER",ITER,loss)
        #print(loss / chars,)
        #print "TM:",(time.time() - _start)/len(train)
        sgd.status()
        sgd.update_epoch(1.0)
