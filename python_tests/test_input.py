from __future__ import print_function
import dynet as dy
import numpy as np

input_vals = np.arange(81)
squared_norm = (input_vals**2).sum()
shapes = [(81,), (3, 27), (3, 3, 9), (3, 3, 3, 3)]
for i in range(4):
    # Not batched
    dy.renew_cg()
    input_tensor = input_vals.reshape(shapes[i])
    x = dy.inputTensor(input_tensor)
    assert (x.dim()[:-1] == shapes[i] and x.dim()[-1] == 1),"Dimension mismatch : {} : {}, {} : 1".format(x.dim()[:-1], shapes[i], x.dim()[-1])
    assert (x.npvalue() == input_tensor).all(), "Expression value different from initial value"
    assert dy.squared_norm(x).scalar_value() == squared_norm, "Value mismatch"
    # Batched
    dy.renew_cg()
    xb = dy.inputTensor(input_tensor, batched=True)
    assert (xb.dim() == shapes[i] or (i == 0 and xb.dim() == (1,shapes[0][0]))), "Dimension mismatch with batch size"
    assert (xb.npvalue() == input_tensor).all(), "Batched expression value different from initial value"
    assert dy.sum_batches(dy.squared_norm(xb)).scalar_value() == squared_norm, "Value mismatch"

caught = False
try:
    dy.renew_cg()
    x = dy.inputTensor("This is not a tensor", batched=True)
except TypeError:
    caught = True

assert caught, "Exception wasn't caught"
