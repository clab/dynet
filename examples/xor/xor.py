import sys
import dynet as dy

#xsent = True
xsent = False

HIDDEN_SIZE = 8
ITERATIONS = 2000

m = dy.Model()
trainer = dy.SimpleSGDTrainer(m)

W = m.add_parameters((HIDDEN_SIZE, 2))
b = m.add_parameters(HIDDEN_SIZE)
V = m.add_parameters((1, HIDDEN_SIZE))
a = m.add_parameters(1)

if len(sys.argv) == 2:
  m.populate_from_textfile(sys.argv[1])

x = dy.vecInput(2)
y = dy.scalarInput(0)
h = dy.tanh((W*x) + b)
if xsent:
    y_pred = dy.logistic((V*h) + a)
    loss = dy.binary_log_loss(y_pred, y)
    T = 1
    F = 0
else:
    y_pred = (V*h) + a
    loss = dy.squared_distance(y_pred, y)
    T = 1
    F = -1


for iter in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi // 2) % 2
        x.set([T if x1 else F, T if x2 else F])
        y.set(T if x1 != x2 else F)
        mloss += loss.scalar_value()
        loss.backward()
        trainer.update()
    mloss /= 4.
    print("loss: %0.9f" % mloss)

x.set([F,T])
z = -(-y_pred)
print(z.scalar_value())

m.save("xor.pymodel")

dy.renew_cg()

x = dy.vecInput(2)
y = dy.scalarInput(0)
h = dy.tanh((W*x) + b)
if xsent:
    y_pred = dy.logistic((V*h) + a)
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

