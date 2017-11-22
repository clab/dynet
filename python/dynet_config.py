def set(mem="512", random_seed=0, autobatch=0,
        profiling=0, weight_decay=0, shared_parameters=0,
        requested_gpus=0, gpu_mask=None):

    if "__DYNET_CONFIG" in __builtins__:
        (mem, random_seed, auto_batch, profiling) = (
            __builtins__["__DYNET_CONFIG"]["mem"] if __builtins__["__DYNET_CONFIG"].get("mem") == mem else mem,
            __builtins__["__DYNET_CONFIG"]["seed"] if __builtins__["__DYNET_CONFIG"].get("seed") == random_seed else random_seed,
            __builtins__["__DYNET_CONFIG"]["autobatch"] if __builtins__["__DYNET_CONFIG"].get("autobatch") == autobatch else autobatch,
            __builtins__["__DYNET_CONFIG"]["profiling"] if __builtins__["__DYNET_CONFIG"].get("profiling") == profiling else profiling)
        (weight_decay, shared_parameters, requested_gpus, gpu_mask) = (
            __builtins__["__DYNET_CONFIG"]["weight_decay"] if __builtins__["__DYNET_CONFIG"].get("weight_decay") == weight_decay else weight_decay,
            __builtins__["__DYNET_CONFIG"]["shared_params"] if __builtins__["__DYNET_CONFIG"].get("shared_params") == shared_parameters else shared_parameters,
            __builtins__["__DYNET_CONFIG"]["requested_gpus"] if __builtins__["__DYNET_CONFIG"].get("requested_gpus") == requested_gpus else requested_gpus,
            __builtins__["__DYNET_CONFIG"]["gpu_mask"] if __builtins__["__DYNET_CONFIG"].get("gpu_mask") == gpu_mask else gpu_mask)

    # TODO read "gpu_mask" from list of IDs?
    __builtins__["__DYNET_CONFIG"] = {
      "mem":mem, "seed": random_seed, "autobatch": autobatch,
      "profiling":profiling, "weight_decay": weight_decay,
      "shared_params": shared_parameters,
      "requested_gpus": requested_gpus,
      "gpu_mask": gpu_mask if gpu_mask else list(),
    }

def set_gpu(flag=True):
    __builtins__["__DYNET_GPU"]=flag
    if "__DYNET_CONFIG" in __builtins__:
        __builtins__["__DYNET_CONFIG"]["requested_gpus"] = 1
    else:
        set(requested_gpus=1)

def gpu():
    if "__DYNET_GPU" in __builtins__:
        return __builtins__["__DYNET_GPU"]
    return None

def get():
    if "__DYNET_CONFIG" in __builtins__:
        return __builtins__["__DYNET_CONFIG"]
    return None
