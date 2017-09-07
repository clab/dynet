import dynet as dy
import numpy as np

m = dy.Model()
lp = m.add_lookup_parameters((100,10))

# regular lookup
a = lp[1].npvalue()
b = lp[2].npvalue()
c = lp[3].npvalue()

# batch lookup instead of single elements.
# two ways of doing this.
abc1 = dy.lookup_batch(lp, [1,2,3])
print(abc1.npvalue())

abc2 = lp.batch([1,2,3])
print(abc2.npvalue())

print(np.hstack([a,b,c]))


# use pick and pickneglogsoftmax in batch mode
# (must be used in conjunction with lookup_batch):
print("\nPick")
W = dy.parameter( m.add_parameters((5, 10)) )
h = W * lp.batch([1,2,3])
print(h.npvalue())
print(dy.pick_batch(h,[1,2,3]).npvalue())
print(dy.pick(W*lp[1],1).value(), dy.pick(W*lp[2],2).value(), dy.pick(W*lp[3],3).value())

# using pickneglogsoftmax_batch
print("\nPick neg log softmax")
print((-dy.log(dy.softmax(h))).npvalue())
print(dy.pickneglogsoftmax_batch(h,[1,2,3]).npvalue())
