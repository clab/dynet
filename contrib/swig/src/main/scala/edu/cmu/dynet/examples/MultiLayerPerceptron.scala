package edu.cmu.dynet.examples

import edu.cmu.dynet._

import scala.language.implicitConversions

object Activation extends Enumeration {
  type Activation = Value
  val SIGMOID, TANH, RELU, LINEAR, SOFTMAX = Value
}

case class Layer(
  inputDim: Int = 0,
  outputDim: Int = 0,
  activation: Activation.Value = Activation.LINEAR,
  dropoutRate: Float = 0
) {}

case class LayerParams(w: Parameter, b: Parameter)

class MultiLayerPerceptron(model: ParameterCollection, layers: Seq[Layer]) {
  // check that layers are compatible
  layers.sliding(2).foreach(pair => assert(pair(0).outputDim == pair(1).inputDim))

  var dropoutActive = true

  // Add parameters to model
  val params = for {
    layer <- layers
    w = model.addParameters(Dim(layer.outputDim, layer.inputDim))
    b = model.addParameters(Dim(layer.outputDim))
  } yield LayerParams(w, b)

  def run(x: Expression): Expression = {
    // expression for the current hidden state
    var h_cur = x
    for ((layer, layerParams) <- layers.zip(params)) {
      // initialize parameters in computation graph
      val W = Expression.parameter(layerParams.w)
      val b = Expression.parameter(layerParams.b)
      // apply affine transform
      val ev = new ExpressionVector(Seq(b, W, h_cur))
      val a = Expression.affineTransform(ev)
      // apply activation function
      val h = activate(a, layer.activation)
      // take care of dropout
      val h_dropped = if (layer.dropoutRate > 0) {
        if (dropoutActive) {
          // during training, drop random units
          val mask = Expression.randomBernoulli(Dim(layer.outputDim), 1 - layer.dropoutRate)
          Expression.cmult(h, mask)
        } else {
          // at test time, multiply by the retention rate to scale
          h * (1 - layer.dropoutRate)
        }
      } else {
        // no dropout, don't do anything
        h
      }
      // Set current hidden state
      h_cur = h_dropped
    }
    h_cur
  }

  def get_nll(x: Expression, labels: UnsignedVector): Expression = {
    // compute output
    val y = run(x)
    // do softmax
    val losses = Expression.pickNegLogSoftmax(y, labels)
    // sum across batches
    val result = Expression.sumBatches(losses)
    result
  }

  def predict(x: Expression): Int = {
    // run MLP to get class distribution
    val y = run(x)
    // get values
    val probs = ComputationGraph.forward(y).toSeq
    // return the argmax
    probs.zipWithIndex.max._2
  }

  def enable_dropout(): Unit = { dropoutActive = true }

  def disable_dropout(): Unit = { dropoutActive = false }

  def is_dropout_enabled(): Boolean = dropoutActive

  def activate(h: Expression, f: Activation.Value): Expression = f match {
    case Activation.LINEAR => h
    case Activation.RELU => Expression.rectify(h)
    case Activation.SIGMOID => Expression.logistic(h)
    case Activation.TANH => Expression.tanh(h)
    case Activation.SOFTMAX => Expression.softmax(h)
    case _ => throw new IllegalArgumentException("unknown activation function")
  }
}
