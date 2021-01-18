package edu.cmu.dynet

/** Contains methods for initializing DyNet */
object Initialize {
  private def initialize(params: internal.DynetParams): Unit = {
    internal.dynet_swig.initialize(params)
  }

  /** Initializes DyNet using the specified args
    *
    * @param args a Map of key-value pairs indicating any args for initialization
    */
  def initialize(args: Map[String, Any] = Map.empty): Unit = {
    val params = new internal.DynetParams()

    args.get("dynet-mem")
        .foreach(arg => params.setMem_descriptor(arg.asInstanceOf[String]))

    args.get("random-seed")
        .foreach(arg => params.setRandom_seed(arg.asInstanceOf[Long]))

    args.get("weight-decay")
        .foreach(arg => params.setWeight_decay(arg.asInstanceOf[Float]))

    args.get("shared-parameters")
        .foreach(arg => params.setShared_parameters(arg.asInstanceOf[Boolean]))

    args.get("autobatch")
        .foreach(arg => params.setAutobatch(arg.asInstanceOf[Int]))

    args.get("profiling")
      .foreach(arg => params.setProfiling(arg.asInstanceOf[Int]))

    args.get("gpus")
      .foreach(arg => params.setRequested_gpus(arg.asInstanceOf[Int]))

    if(args.contains("devices")){
      require(!params.getIds_requested)
      params.setIds_requested(true)
      args.get("devices")
        .foreach(arg => arg.asInstanceOf[String].split(',').foreach(
          s =>
            if(s.startsWith("CPU:")){
              Console.err.println("DyNet doesn't support specifying CPU id")
            }else if(s.startsWith("GPU:")){
              val gpuID = s.split(":")(1).toInt
              params.getGpu_mask.set(gpuID, params.getGpu_mask.get(gpuID) + 1)
              params.setRequested_gpus(params.getRequested_gpus + 1)
              require(params.getGpu_mask.get(gpuID) == 1)
            }
        ))
    }

    initialize(params)
  }
}