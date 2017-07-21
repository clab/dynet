def set(mem="0", random_seed=0, autobatch=0,
        autobatch_debug=0, weight_decay=0, shared_parameters=0,
        requested_gpus=0, gpu_mask=None):

    # TODO read "gpu_mask" from list of IDs?
    __builtins__["__DYNET_CONFIG"] = {
            "mem":mem, "seed": random_seed, "autobatch": autobatch,
            "autobatch_debug":autobatch_debug, "weight_decay": weight_decay,
            "shared_params": shared_parameters,
            "requested_gpus": requested_gpus,
            "gpu_mask": gpu_mask if gpu_mask else list(),
            }

def set_gpu(flag=True):
    __builtins__["__DYNET_GPU"]=flag

def gpu(): return __builtins__["__DYNET_GPU"]

def get():
    if "__DYNET_CONFIG" in __builtins__:
        return __builtins__["__DYNET_CONFIG"]
    return None
