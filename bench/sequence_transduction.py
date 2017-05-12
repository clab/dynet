import dynet as dy
import random
import time
import sys
random.seed(1)

SEQ_LENGTH=2
BATCH_SIZE=2
HIDDEN=1
NCLASSS=2
EMBED_SIZE=1
N_SEQS=1000
autobatching=True

dy.renew_cg()

random_seq = lambda ln,t: [random.randint(0,t-1) for _ in xrange(ln)]
seq_lengths = [SEQ_LENGTH for _ in range(N_SEQS)]
#seq_lengths = [random.randint(10, SEQ_LENGTH) for _ in range(N_SEQS)]
Xs = [random_seq(L, 100) for L in seq_lengths]
Ys = [random_seq(L, NCLASSS) for L in seq_lengths]

m = dy.Model()
trainer = dy.SimpleSGDTrainer(m)

E = m.add_lookup_parameters((1000, EMBED_SIZE))
fwR = dy.VanillaLSTMBuilder(1, EMBED_SIZE, HIDDEN, m)
bwR = dy.VanillaLSTMBuilder(1, EMBED_SIZE, HIDDEN, m)
T_= m.add_parameters((HIDDEN, HIDDEN*2))
fwR2 = dy.VanillaLSTMBuilder(1, EMBED_SIZE, HIDDEN, m)
bwR2 = dy.VanillaLSTMBuilder(1, EMBED_SIZE, HIDDEN, m)
W_= m.add_parameters((NCLASSS, HIDDEN*1))

total_time = 0.0

def transduce(seq,Y):
    seq = [E[i] for i in seq]
    fw = fwR.initial_state().transduce(seq)

    # this UNUSED part affects strategy 2
    XXX = fwR2.initial_state().transduce([E[3],E[5]])

    W = W_.expr()
    outs = [W*z for z in fw]
    losses = [dy.pickneglogsoftmax(o,y) for o,y in zip(outs,Y)]
    s = dy.esum(losses)
    return s

batch=[]
start = time.time()
for X,Y in zip(Xs,Ys):
    loss = transduce(X,Y)
    batch.append(loss)
    if len(batch)==BATCH_SIZE:
        s = dy.esum(batch)
        s_ = time.time()
        s.forward()
        total_time = total_time + time.time() - s_
        print s.npvalue()
        sys.exit()
        #break
        s.backward()
        trainer.update()
        batch = []
        dy.renew_cg()
print "total time:",time.time() - start, len(Xs) / (time.time() - start)
print "forward time:",total_time, len(Xs) / total_time
    


