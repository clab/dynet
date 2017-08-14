import sys
if '--dynet-viz' in sys.argv:
    sys.argv.remove('--dynet-viz')
    from dynet_viz import *
elif '--dynet-gpu' in sys.argv: # the python gpu switch, use GPU:0 by defailt
    sys.argv.remove('--dynet-gpu')
    def print_graphviz(**kwarge):
        print("Run with --dynet-viz to get the visualization behavior.")
    sys.argv.append('--dynet-devices')
    sys.argv.append('GPU:0')
    from _dynet import *
elif '--dynet-gpus' in sys.argv or '--dynet-devices' in sys.argv: # but using the c++ gpu switches suffices to trigger gpu.
    def print_graphviz(**kwarge):
        print("Run with --dynet-viz to get the visualization behavior.")
    from _dynet import *
else: # use CPU by default
    sys.argv.append('--dynet-devices')
    sys.argv.append('CPU')
    def print_graphviz(**kwarge):
        print("Run with --dynet-viz to get the visualization behavior.")
    from _dynet import *

__version__ = 2.0

init()
