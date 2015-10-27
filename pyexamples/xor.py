from pycnn import *

#xsent = True
xsent = False

HIDDEN_SIZE = 8
ITERATIONS = 2000

m = Model()
sgd = SimpleSGDTrainer(m)

m.add_parameters("W",(HIDDEN_SIZE, 2))
m.add_parameters("b",HIDDEN_SIZE)
m.add_parameters("V",(1, HIDDEN_SIZE))
m.add_parameters("a",1)

W = parameter(m["W"])
b = parameter(m["b"])
V = parameter(m["V"])
a = parameter(m["a"])

x = vecInput(2)
y = scalarInput(0)
h = tanh((W*x) + b)
if xsent:
    y_pred = logistic((V*h) + a)
    loss = binary_log_loss(y_pred, y)
    T = 1
    F = 0
else:
    y_pred = (V*h) + a
    loss = squared_distance(y_pred, y)
    T = 1
    F = -1


for iter in xrange(ITERATIONS):
    mloss = 0.0
    for mi in xrange(4):
        x1 = mi % 2
        x2 = (mi / 2) % 2
        x.set([T if x1 else F, T if x2 else F])
        y.set(T if x1 != x2 else F)
        #mloss += cg().forward_scalar()
        mloss += loss.scalar_value()
        #cg().backward()
        loss.backward()
        sgd.update(1.0)
    sgd.update_epoch();
    mloss /= 4.
    print "loss: %0.9f" % mloss

x.set([F,T])
z = -(-y_pred)
print z.scalar_value()
#print y_pred.scalar()

renew_cg()
W = parameter(m["W"])
b = parameter(m["b"])
V = parameter(m["V"])
a = parameter(m["a"])

x = vecInput(2)
y = scalarInput(0)
h = tanh((W*x) + b)
if xsent:
    y_pred = logistic((V*h) + a)
else:
    y_pred = (V*h) + a
x.set([T,F])
print "TF",y_pred.scalar_value()
x.set([F,F])
print "FF",y_pred.scalar_value()
x.set([T,T])
print "TT",y_pred.scalar_value()
x.set([F,T])
print "FT",y_pred.scalar_value()

