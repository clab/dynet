import dynet as dy
import numpy as np

m = dy.Model()
p = m.add_lookup_parameters((2,3))
npp = np.asarray([[1,2,3],[4,5,6]], dtype=np.float32)
p.init_from_array(npp)
dy.renew_cg()

x = dy.lookup_batch(p, [0,1])
y = dy.pick_batch_elems(x, [0])
z = dy.pick_batch_elem(x, 1)
yz = dy.pick_batch_elems(x, [0, 1])
w = dy.concat_to_batch([y,z])

print x.npvalue()
print y.npvalue()
print yz.npvalue()
print w.npvalue()

loss = dy.dot_product(y, z)
loss.forward()

loss.backward()

print p.grad_as_array()
