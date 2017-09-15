package edu.cmu.dynet

/** Builder method for creating GRUs, as in the C++ code. For its public methods see
  * [[edu.cmu.dynet.RnnBuilder]].
  */
class GruBuilder private[dynet](private[dynet] val builder: internal.GRUBuilder)
    extends RnnBuilder(builder) {

  /** Create a new, empty GruBuilder. */
  def this() { this(new internal.GRUBuilder()) }

  /** Create a GruBuilder with the specified parameters.
    */
  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection) {
    this(new internal.GRUBuilder(layers, inputDim, hiddenDim, model.model))
  }
}

