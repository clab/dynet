import dynet as dy
import numpy as np

# create model
m=dy.Model()
# add parameter
p=m.parameters_from_numpy(np.ones((1,5)))
# create cg
dy.renew_cg()
# input tensor
x= dy.inputTensor(np.arange(5).reshape((5,1)))
# add parameter to computation graph
e_p=dy.parameter(p)
# compute dot product
res = e_p * x
# Run forward and backward pass
res.forward()
res.backward()
# Should print the value of x
print p.grad_as_array()
