## TODO: make into a proper test.
## TODO: test saving and loading.
import dynet as dy

m = dy.Model()
trainer = dy.AdamTrainer(m)
trainer = dy.SimpleSGDTrainer(m)

p1 = m.add_parameters((10,10))
p2 = m.add_parameters((10,10))
lp1 = m.add_lookup_parameters((10,10))
lp2 = m.add_lookup_parameters((10,10))


assert( p1.is_updatable() )
assert( p2.is_updatable() )
assert( lp1.is_updatable() )
assert( lp2.is_updatable() )

p2.set_update(False)
lp1.set_update(False)

assert (p1.is_updatable())
assert (not p2.is_updatable())
assert (not lp1.is_updatable())
assert ( lp2.is_updatable() )

p1.set_update(True)
p2.set_update(False)
lp1.set_update(False)
lp2.set_update(True)

assert (p1.is_updatable())
assert (not p2.is_updatable())
assert (not lp1.is_updatable())
assert ( lp2.is_updatable() )

p1.set_update(False)
p2.set_update(True)
lp1.set_update(True)
lp2.set_update(False)

assert (not p1.is_updatable())
assert (p2.is_updatable())
assert (lp1.is_updatable())
assert (not lp2.is_updatable() )

import numpy as np
x = np.ones((10,10))

p1.load_array(x)
p2.load_array(x)
lp1.init_from_array(x)
lp2.init_from_array(x)


pp1 = dy.parameter(p1)
pp2 = dy.parameter(p2)

a = pp1 * lp1[1]
b = pp2 * lp2[1]
l = dy.dot_product(a,b)
l.npvalue()
l.backward()
trainer.update()

print p1.as_array() # ones, no updates occured.
print p2.as_array() # 0.99, updates did occur
print lp1.as_array() # 0.99, update did occur
print lp2.as_array() # 1.0, not update occured

dy.renew_cg()
p1.set_update(True)
p2.set_update(True)
lp1.set_update(True)
lp2.set_update(True)

pp1 = dy.parameter(p1)
pp2 = dy.parameter(p2)

a = pp1 * lp1[1]
b = pp2 * lp2[1]
l = dy.dot_product(a,b)
l.npvalue()
l.backward()
trainer.update()

print
print p1.as_array() # 0.99
print p2.as_array() # below 0.99, updates did occur
print lp1.as_array() # below 0.99, update did occur
print lp2.as_array() # 0.99


