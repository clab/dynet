package edu.cmu.dynet

class VanillaLstmBuilder private[dynet](private[dynet] val builder: internal.VanillaLSTMBuilder)
  extends RnnBuilder(builder) {

  def this() { this(new internal.VanillaLSTMBuilder()) }

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: Model) {
    this(new internal.VanillaLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }
}

/** Builder method for creating LSTMs, as in the C++ code. For its public methods see
  * [[edu.cmu.dynet.RnnBuilder]].
  */
class LstmBuilder private[dynet](private[dynet] val builder: internal.LSTMBuilder)
    extends RnnBuilder(builder) {

  /** Create a new, empty LstmBuilder. */
  def this() { this(new internal.LSTMBuilder()) }

  /** Create a LstmBuilder with the specified parameters.
    */
  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: Model) {
    this(new internal.LSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }
}


