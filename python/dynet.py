import sys
if '--dynet-viz' in sys.argv:
    from dynet_viz import *
elif '--dynet-gpu' in sys.argv:
    def print_graphviz(**kwarge):
        print "Run with --dynet-viz to get the visualization behavior."
    from _gdynet import *
else:
    def print_graphviz(**kwarge):
        print "Run with --dynet-viz to get the visualization behavior."
    from _dynet import *

__version__ = 2.0
