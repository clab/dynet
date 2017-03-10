package edu.cmu.dynet

object Initialize {
  private def initialize(params: internal.DynetParams): Unit = {
    internal.dynet_swig.initialize(params)
  }

  def initialize(args: Map[String, String] = Map.empty): Unit = {
    val params = new internal.DynetParams()

    args.get("dynet-mem") match {
      case Some(mem) => params.setMem_descriptor(mem)
      case None => ()
    }

    initialize(params)
  }
}