from dynet import *

#xsent = True
xsent = False

HIDDEN_SIZE = 8
ITERATIONS = 2000

m = Model()
sgd = SimpleSGDTrainer(m)

pW = m.add_parameters((HIDDEN_SIZE, 2))
pb = m.add_parameters(HIDDEN_SIZE)
pV = m.add_parameters((1, HIDDEN_SIZE))
pa = m.add_parameters(1)

W = parameter(pW)
b = parameter(pb)
V = parameter(pV)
a = parameter(pa)

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


for iter in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi / 2) % 2
        x.set([T if x1 else F, T if x2 else F])
        y.set(T if x1 != x2 else F)
        mloss += loss.scalar_value()
        loss.backward()
        sgd.update(1.0)
    sgd.update_epoch();
    mloss /= 4.
    print("loss: %0.9f" % mloss)

x.set([F,T])
z = -(-y_pred)
print(z.scalar_value())

renew_cg()
W = parameter(pW)
b = parameter(pb)
V = parameter(pV)
a = parameter(pa)

x = vecInput(2)
y = scalarInput(0)
h = tanh((W*x) + b)
if xsent:
    y_pred = logistic((V*h) + a)
else:
    y_pred = (V*h) + a
x.set([T,F])
print("TF",y_pred.scalar_value())
x.set([F,F])
print("FF",y_pred.scalar_value())
x.set([T,T])
print("TT",y_pred.scalar_value())
x.set([F,T])
print("FT",y_pred.scalar_value())

