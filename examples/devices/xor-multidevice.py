# Usage:
# python xor-multidevice.py --dynet-devices CPU,GPU:0,GPU:1
# or python xor-multidevice.py --dynet-gpus 2

import sys 
import dynet as dy


#xsent = True
xsent = False

HIDDEN_SIZE = 8 
ITERATIONS = 2000

m = dy.Model()
trainer = dy.SimpleSGDTrainer(m)

pW1 = m.add_parameters((HIDDEN_SIZE, 2), device="GPU:1")
pb1 = m.add_parameters(HIDDEN_SIZE, device="GPU:1")
pW2 = m.add_parameters((HIDDEN_SIZE, HIDDEN_SIZE), device="GPU:0")
pb2 = m.add_parameters(HIDDEN_SIZE, device="GPU:0")
pV = m.add_parameters((1, HIDDEN_SIZE), device="CPU")
pa = m.add_parameters(1, device="CPU")

if len(sys.argv) == 2:
  m.populate_from_textfile(sys.argv[1])

dy.renew_cg()
W1, b1, W2, b2, V, a = dy.parameter(pW1, pb1, pW2, pb2, pV, pa)

x = dy.vecInput(2, "GPU:1")
y = dy.scalarInput(0, "CPU")
h1 = dy.tanh((W1*x) + b1)
h1_gpu0 = dy.to_device(h1, "GPU:0")
h2 = dy.tanh((W2*h1_gpu0) + b2)
h2_cpu = dy.to_device(h2, "CPU")
if xsent:
    y_pred = dy.logistic((V*h2_cpu) + a)
    loss = dy.binary_log_loss(y_pred, y)
    T = 1 
    F = 0 
else:
    y_pred = (V*h2_cpu) + a 
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
W1, b1, W2, b2, V, a = dy.parameter(pW1, pb1, pW2, pb2, pV, pa)

x = dy.vecInput(2, "GPU:1")
y = dy.scalarInput(0, "CPU")
h1 = dy.tanh((W1*x) + b1)
h1_gpu0 = dy.to_device(h1, "GPU:0")
h2 = dy.tanh((W2*h1_gpu0) + b2)
h2_cpu = dy.to_device(h2, "CPU")
if xsent:
    y_pred = dy.logistic((V*h2_cpu) + a)
else:
    y_pred = (V*h2_cpu) + a 
x.set([T,F])
print("TF",y_pred.scalar_value())
x.set([F,F])
print("FF",y_pred.scalar_value())
x.set([T,T])
print("TT",y_pred.scalar_value())
x.set([F,T])
print("FT",y_pred.scalar_value())
