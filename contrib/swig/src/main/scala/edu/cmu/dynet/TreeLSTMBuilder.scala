package edu.cmu.dynet


abstract class TreeLSTMBuilder[A<: internal.TreeLSTMBuilder] private[dynet](private[dynet] val builder: A)
  extends RnnBuilder(builder) {

  def addInput(id: Int, children: IntVector, x: Expression): Expression = {
    val expr = builder.add_input(id, children.vector, x.expr)
    new Expression(expr)
  }

  def setNumElements(num: Int) = {
    builder.set_num_elements(num)
  }
}


class BidirectionalTreeLSTMBuilder private[dynet](private[dynet] builder: internal.BidirectionalTreeLSTMBuilder)
  extends TreeLSTMBuilder[internal.BidirectionalTreeLSTMBuilder](builder) {

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection) {
    this(new internal.BidirectionalTreeLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }
}

class UnidirectionalTreeLSTMBuilder private[dynet](private[dynet] builder: internal.UnidirectionalTreeLSTMBuilder)
  extends TreeLSTMBuilder[internal.UnidirectionalTreeLSTMBuilder](builder) {

  def this(layers: Long, inputDim: Long, hiddenDim: Long, model: ParameterCollection) {
    this(new internal.UnidirectionalTreeLSTMBuilder(layers, inputDim, hiddenDim, model.model))
  }
}
