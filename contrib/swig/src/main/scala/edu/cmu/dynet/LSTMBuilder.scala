package edu.cmu.dynet

class VanillaLstmBuilder private[dynet](private[dynet] val builder: internal.VanillaLSTMBuilder)
  extends RnnBuilder(builder) {

  def this() { this(new internal.VanillaLSTMBuilder()) }

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection, lnLSTM: Boolean = false) {
    this(new internal.VanillaLSTMBuilder(layers, inputDim, hiddenDim, model.model, lnLSTM))
  }

  def setDropout(d: Float, dR: Float): Unit = builder.set_dropout(d, dR)

  def setDropoutMasks(batchSize:Long): Unit = builder.set_dropout_masks(batchSize)
}

// TODO(joelgrus): get the typedef to work
class LstmBuilder private[dynet](private[dynet] val builder: internal.VanillaLSTMBuilder)
  extends RnnBuilder(builder) {

  def this() { this(new internal.VanillaLSTMBuilder()) }

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection, lnLSTM: Boolean = false) {
    this(new internal.VanillaLSTMBuilder(layers, inputDim, hiddenDim, model.model, lnLSTM))
  }

  def setDropout(d: Float, dR: Float): Unit = builder.set_dropout(d, dR)

  def setDropoutMasks(batchSize:Long): Unit = builder.set_dropout_masks(batchSize)
}

/** Builder method for creating LSTMs, as in the C++ code. For its public methods see
  * [[edu.cmu.dynet.RnnBuilder]].
  */
class CoupledLstmBuilder private[dynet](private[dynet] val builder: internal.CoupledLSTMBuilder)
  extends RnnBuilder(builder) {

  /** Create a new, empty LstmBuilder. */
  def this() { this(new internal.CoupledLSTMBuilder()) }

  /** Create a LstmBuilder with the specified parameters.
    */
  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection) {
    this(new internal.CoupledLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }

  def setDropout(d: Float, dH: Float, dC: Float): Unit = builder.set_dropout(d, dH, dC)

  def setDropoutMasks(batchSize:Long): Unit = builder.set_dropout_masks(batchSize)
}

class CompactVanillaLSTMBuilder private[dynet](private[dynet] val builder: internal.CompactVanillaLSTMBuilder)
  extends RnnBuilder(builder) {

  def this() {this(new internal.CompactVanillaLSTMBuilder()) }

  def this(layers:Long, inputDim:Long, hiddenDim: Long, model: ParameterCollection) {
    this(new internal.CompactVanillaLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }

  def setDropout(d: Float, dR: Float): Unit = builder.set_dropout(d, dR)

  def setDropoutMasks(batchSize:Long): Unit = builder.set_dropout_masks(batchSize)
}
