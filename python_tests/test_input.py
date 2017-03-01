from __future__ import print_function
import dynet as dy
import numpy as np

input_vals = np.arange(81)
shapes=[(81),(3,27),(3,3,9)]
for i in range(3):
    # Not batched
    dy.renew_cg()
    input_tensor=input_vals.reshape(shapes[i])
    x=dy.inputTensor(input_tensor)
    print(x.dim())
    print(x.npvalue())
    print(dy.squared_norm(x).npvalue())
    # Batched
    dy.renew_cg()
    xb=dy.inputTensor(input_tensor,batched=True)
    print(xb.dim())
    print(xb.npvalue())
    print(dy.sum_batches(dy.squared_norm(xb)).npvalue())

caught = False
try:
    dy.renew_cg()
    x=dy.inputTensor("This is not a tensor",batched=True)
except TypeError:
    caught=True

assert(caught,"Exception wasn't caught")
