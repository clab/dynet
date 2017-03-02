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
    assert (x.dim()[0] == shapes[i] and x.dim()[1] == 1),"Dimension mismatch : {} : ({}, {})".format(x.dim(), shapes[i],1)
    assert (x.npvalue() == input_tensor).all(), "Expression value different from initial value"
    assert dy.squared_norm(x).scalar_value() == squared_norm, "Value mismatch"
    # Batched
    dy.renew_cg()
    xb = dy.inputTensor(input_tensor, batched=True)
    assert (xb.dim()[0] == (shapes[i][:-1] if i>0 else (1,)) and xb.dim()[1] == shapes[i][-1]), "Dimension mismatch with batch size : {} : ({}, {})".format(xb.dim(), (shapes[i][:-1] if i>0 else 1),shapes[i][-1])
    assert (xb.npvalue() == input_tensor).all(), "Batched expression value different from initial value"
    assert dy.sum_batches(dy.squared_norm(xb)).scalar_value() == squared_norm, "Value mismatch"
    # Batched with list
    dy.renew_cg()
    xb = dy.inputTensor([np.asarray(x).transpose() for x in input_tensor.transpose()])
    assert (xb.dim()[0] == (shapes[i][:-1] if i>0 else (1,)) and xb.dim()[1] == shapes[i][-1]) , "Dimension mismatch with batch size : {} : ({}, {})".format(xb.dim(), (shapes[i][:-1] if i>0 else 1),shapes[i][-1])
    assert (xb.npvalue() == input_tensor).all(), "Batched expression value different from initial value"
    assert dy.sum_batches(dy.squared_norm(xb)).scalar_value() == squared_norm, "Value mismatch"

caught = False
try:
    dy.renew_cg()
    x = dy.inputTensor("This is not a tensor", batched=True)
except TypeError:
    caught = True

assert caught, "Exception wasn't caught"
