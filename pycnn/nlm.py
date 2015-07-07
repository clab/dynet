from pycnn import *



from util import CorpusReader, Vocab
#corpus = CorpusReader("../examples/example-data/fin-toy.txt")
corpus = CorpusReader("ngrams")
vocab = Vocab.from_corpus(corpus)

def corpus_to_examples(corpus,CONTEXT=3):
    for sent in corpus:
        for i in xrange(CONTEXT,len(sent)):
            ctx = sent[i-CONTEXT:i]
            word = sent[i]
            yield (ctx, word)

#for ctx, word in corpus_to_examples(corpus):
#    print " ".join(str(vocab.w2i[c]) for c in ctx),vocab.w2i[word]
#sys.exit()


print vocab.size()

CONTEXT = 3
DIM = 100
VOCAB_SIZE = vocab.size()

m = Model()
sgd = SimpleSGDTrainer(m)
cg = ComputationGraph()

class InputWordIds:
    def __init__(self, cg, lookup_params, nwords):
        self.exprs = [LookupExpression(cg, lookup_params) for _ in xrange(nwords)]
    def feed(self, word_ids):
        [e.set(i) for (e,i) in zip(self.exprs,word_ids)]

# TODO this can be made nicer I think
m.add_lookup_parameters("word_lookup", (VOCAB_SIZE, DIM))
contexts = InputWordIds(cg, m["word_lookup"], CONTEXT)
cvec = concatenate(contexts.exprs) # TODO

C = cg.parameters(m, "C", (DIM, DIM*CONTEXT))
hb = cg.parameters(m, "hb", DIM)
R = cg.parameters(m, "R", (VOCAB_SIZE, DIM))
bias = cg.parameters(m, "bias", VOCAB_SIZE)

r = hb + (C * cvec)
nl = rectify(r)
o2 = bias + (R * nl)
ydist = log_softmax(o2)
expected_outcome = IntValue()
nerr = -pick(ydist, expected_outcome) # TODO: make nicer API!

cg.PrintGraphviz()


data = list(corpus_to_examples(corpus))
loss = 0.0
#for i, (context_words, target_word) in enumerate(corpus_to_examples(corpus)):
import time
thestart = time.time()
for iter in xrange(100):
    start = time.time()
    for i, (context_words, target_word) in enumerate(data):
        contexts.feed([vocab.w2i[w] for w in context_words])
        expected_outcome.set(vocab.w2i[target_word])
        loss += cg.forward_scalar()
        cg.backward()
        sgd.update(1.0)
        if i == 2500: break
    print loss / float(i)
    loss = 0.0
    print "TIME:", (time.time() - start) * 1000

print "total:",time.time() - thestart
