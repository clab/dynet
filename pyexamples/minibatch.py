from pycnn import *
import numpy as np

m = Model()
lp = m.add_lookup_parameters("a",(100,10))

# regular lookup
a = lp[1].npvalue()
b = lp[2].npvalue()
c = lp[3].npvalue()

# batch lookup instead of single elements.
# two ways of doing this.
abc1 = lookup_batch(lp, [1,2,3])
print abc1.npvalue()

abc2 = lp.batch([1,2,3])
print abc2.npvalue()

print np.hstack([a,b,c])


# use pick and pickneglogsoftmax in batch mode
# (must be used in conjunction with lookup_batch):
print "\nPick"
W = parameter( m.add_parameters("W", (5, 10)) )
h = W * lp.batch([1,2,3])
print h.npvalue()
print pick_batch(h,[1,2,3]).npvalue()
print pick(W*lp[1],1).value(), pick(W*lp[2],2).value(), pick(W*lp[3],3).value()

# using pickneglogsoftmax_batch
print "\nPick neg log softmax"
print (-log(softmax(h))).npvalue()
print pickneglogsoftmax_batch(h,[1,2,3]).npvalue()
