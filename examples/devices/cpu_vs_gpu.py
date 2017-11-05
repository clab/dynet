# Usage: python cpu_vs_gpu.py

import time
from multiprocessing import Process

def do_cpu():
  import _dynet as C
  C.init()
  cm = C.Model()
  cpW = cm.add_parameters((1000,1000))
  s = time.time()
  C.renew_cg()
  W = C.parameter(cpW)
  W = W*W*W*W*W*W*W
  z = C.squared_distance(W,W)
  z.value()
  z.backward()
  print("CPU time:",time.time() - s)


def do_gpu():
  import _dynet as G
  import sys 
  sys.argv.append('--dynet-devices')
  sys.argv.append('GPU:0')
  G.init()
  gm = G.Model()
  gpW = gm.add_parameters((1000,1000))
  s = time.time()
  G.renew_cg()
  W = G.parameter(gpW)
  W = W*W*W*W*W*W*W
  z = G.squared_distance(W,W)
  z.value()
  z.backward()
  print("GPU time:",time.time() - s)

if __name__ == '__main__':
  procs1 = Process(target=do_cpu)
  procs1.start()
  procs2 = Process(target=do_gpu)
  procs2.start()
  procs1.join()
  procs2.join()
