import sys
if '--dynet-viz' in sys.argv:
    sys.argv.remove('--dynet-viz')
    from dynet_viz import *
else:
    def print_graphviz(**kwarge):
        print("Run with --dynet-viz to get the visualization behavior.")
    from _dynet import *

__version__ = 2.0

init()
