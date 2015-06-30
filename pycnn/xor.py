from pycnn import *

HIDDEN_SIZE = 8
ITERATIONS = 30

m = Model()
sgd = SimpleSGDTrainer(m)

cg = ComputationGraph()

m.add_parameters("W",(HIDDEN_SIZE, 2))
m.add_parameters("b",HIDDEN_SIZE)
m.add_parameters("V",(1, HIDDEN_SIZE))
m.add_parameters("a",1)

W = cg.parameters(m, "W")
b = cg.parameters(m, "b")
V = cg.parameters(m, "V")
a = cg.parameters(m, "a")

x = VecInputExpr2(cg, [0,0])
y = InputExpr2(cg, 0)
#print type(x)
h = tanh((W*x) + b)
y_pred = (V*h) + a
loss = squared_distance(y_pred, y)

for iter in xrange(ITERATIONS):
    mloss = 0.0
    for mi in xrange(4):
        x1 = mi % 2
        x2 = (mi / 2) % 2
        x.set_input([1 if x1 else -1, 1 if x2 else -1])
        y.set_input(1 if x1 != x2 else -1)
        mloss += cg.forward_scalar()
        cg.backward()
        sgd.update(1.0)
    sgd.update_epoch();
    mloss /= 4
    print "loss: %0.9f" % mloss

x.set_input([-1,1])
z = -(-y_pred)
print cg.forward_scalar()
#print y_pred.scalar()

cg = cg.renew()
W = cg.parameters(m, "W")
b = cg.parameters(m, "b")
V = cg.parameters(m, "V")
a = cg.parameters(m, "a")

x = VecInputExpr2(cg, [0,0])
y = InputExpr2(cg, 0)
#print type(x)
h = tanh((W*x) + b)
y_pred = (V*h) + a
x.set_input([1,-1])
print "XX",cg.forward_scalar()
