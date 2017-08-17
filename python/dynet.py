from __future__ import print_function
import sys
import dynet_config

_GPU = False
_CONF = dynet_config.get()

if dynet_config.gpu():
    _GPU=True
# TODO I'm not quite sure what the "use GPU" logic should be.
if _CONF and _CONF["requested_gpus"]:
    _GPU=True

_GPU = _GPU or ("--dynet-gpu" in sys.argv) # the python gpu switch.
_GPU = _GPU or ('--dynet-gpus' in sys.argv or '--dynet-gpu-ids' in sys.argv) # but using the c++ gpu switches suffices to trigger gpu.
if '--dynet-gpu' in sys.argv: sys.argv.remove('--dynet-gpu')

if '--dynet-viz' in sys.argv:
    sys.argv.remove('--dynet-viz')
    from dynet_viz import *
elif _GPU:
    def print_graphviz(**kwarge):
        print("Run with --dynet-viz to get the visualization behavior.")
    print("importing gdy")
    from _gdynet import *
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
