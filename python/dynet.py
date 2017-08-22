from __future__ import print_function
import sys
import dynet_config

_CONF = dynet_config.get()

if '--dynet-viz' in sys.argv:
    sys.argv.remove('--dynet-viz')
    from dynet_viz import *
else:
    def print_graphviz(**kwarge):
        print("Run with --dynet-viz to get the visualization behavior.")
    from _dynet import *

__version__ = 2.0

if _CONF is None:
    init()
else:
    _params = DynetParams()
    _params.from_config(_CONF)
    _params.init()
