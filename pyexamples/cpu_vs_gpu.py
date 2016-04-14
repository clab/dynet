import gpycnn as G
print 
import pycnn as C

cm = C.Model()
gm = G.Model()

cpW = cm.add_parameters("W",(1000,1000))
gpW = gm.add_parameters("W",(1000,1000))

def do_cpu():
	C.renew_cg()
	W = C.parameter(cpW)
	W = W*W*W*W*W*W*W
	z = C.squared_distance(W,W)
	z.value()
	z.backward()

def do_gpu():
	G.renew_cg()
	W = G.parameter(gpW)
	W = W*W*W*W*W*W*W
	z = G.squared_distance(W,W)
	z.value()
	z.backward()

import time
s = time.time()
do_cpu()
print "CPU time:",time.time() - s

s = time.time()
do_gpu()
print "GPU time:",time.time() - s




