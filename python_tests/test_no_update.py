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


assert( p1.is_updated() )
assert( p2.is_updated() )
assert( lp1.is_updated() )
assert( lp2.is_updated() )

p2.set_updated(False)
lp1.set_updated(False)

assert (p1.is_updated())
assert (not p2.is_updated())
assert (not lp1.is_updated())
assert ( lp2.is_updated() )

p1.set_updated(True)
p2.set_updated(False)
lp1.set_updated(False)
lp2.set_updated(True)

assert (p1.is_updated())
assert (not p2.is_updated())
assert (not lp1.is_updated())
assert ( lp2.is_updated() )

p1.set_updated(False)
p2.set_updated(True)
lp1.set_updated(True)
lp2.set_updated(False)

assert (not p1.is_updated())
assert (p2.is_updated())
assert (lp1.is_updated())
assert (not lp2.is_updated() )

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
p1.set_updated(True)
p2.set_updated(True)
lp1.set_updated(True)
lp2.set_updated(True)

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


