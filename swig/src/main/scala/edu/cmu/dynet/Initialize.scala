package edu.cmu.dynet

class DynetParams private[dynet](private[dynet] val params: internal.DynetParams) {
  def this() { this(new internal.DynetParams) }
}

object Initialize {
  def initialize(params: DynetParams): Unit = {
    internal.dynet_swig.initialize(params.params)
  }

  def initialize(): Unit = {
    initialize(new DynetParams)
  }
}