package edu.cmu.dynet

class VanillaLSTMBuilder private[dynet](private[dynet] val builder: internal.VanillaLSTMBuilder)
  extends RNNBuilder(builder) {

  def this() { this(new internal.VanillaLSTMBuilder()) }

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: Model) {
    this(new internal.VanillaLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }
}

/** Builder method for creating LSTMs, as in the C++ code. For its public methods see
  * [[edu.cmu.dynet.RNNBuilder]].
  */
class LSTMBuilder private[dynet](private[dynet] val builder: internal.LSTMBuilder)
    extends RNNBuilder(builder) {

  /** Create a new, empty LSTMBuilder. */
  def this() { this(new internal.LSTMBuilder()) }

  /** Create a LSTMBuilder with the specified parameters.
    */
  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: Model) {
    this(new internal.LSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }
}


