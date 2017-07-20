package edu.cmu.dynet

class FastLstmBuilder private[dynet](private[dynet] val builder: internal.FastLSTMBuilder)
  extends RnnBuilder(builder) {

  def this() { this(new internal.FastLSTMBuilder()) }

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection) {
    this(new internal.FastLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }

}
