package edu.cmu.dynet

/** Builder method for creating GRUs, as in the C++ code. For its public methods see
  * [[edu.cmu.dynet.RNNBuilder]].
  */
class GRUBuilder private[dynet](private[dynet] val builder: internal.GRUBuilder)
    extends RNNBuilder(builder) {

  /** Create a new, empty LSTMBuilder. */
  def this() { this(new internal.GRUBuilder()) }

  /** Create a GRUBuilder with the specified parameters.
    */
  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: Model) {
    this(new internal.GRUBuilder(layers, inputDim, hiddenDim, model.model))
  }
}

