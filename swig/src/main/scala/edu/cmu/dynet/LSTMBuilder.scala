package edu.cmu.dynet

import ImplicitConversions._

class LSTMBuilder private[dynet](builder: internal.LSTMBuilder)
    extends RNNBuilder(builder) {
  // public no-argument constructor
  def this() { this(new internal.LSTMBuilder()) }

  // public many-argument constructor
  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: Model) {
    this(new internal.LSTMBuilder(layers, inputDim, hiddenDim, model))
  }
}
