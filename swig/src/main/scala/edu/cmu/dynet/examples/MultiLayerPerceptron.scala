package edu.cmu.dynet.examples

import edu.cmu.dynet.{dynet_swig => dn, _}

import scala.language.implicitConversions
import DyNetScalaHelpers._

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

class MultiLayerPerceptron(model: Model, layers: Seq[Layer]) {
  // check that layers are compatible
  layers.sliding(2).foreach(pair => assert(pair(0).outputDim == pair(1).inputDim))

  var dropoutActive = true

  // Add parameters to model
  val params = for {
    layer <- layers
    w = model.add_parameters(dim(layer.outputDim, layer.inputDim))
    b = model.add_parameters(dim(layer.outputDim))
  } yield LayerParams(w, b)

  def run(x: Expression, cg: ComputationGraph): Expression = {
    // expression for the current hidden state
    var h_cur = x
    for ((layer, layerParams) <- layers.zip(params)) {
      // initialize parameters in computation graph
      val W = dn.parameter(cg, layerParams.w)
      val b = dn.parameter(cg, layerParams.b)
      // apply affine transform
      val ev = new ExpressionVector(Seq(b, W, h_cur))
      val a = dn.affine_transform(ev)
      // apply activation function
      val h = activate(a, layer.activation)
      // take care of dropout
      val h_dropped = if (layer.dropoutRate > 0) {
        if (dropoutActive) {
          // during training, drop random units
          val mask = dn.random_bernoulli(cg, dim(layer.outputDim), 1 - layer.dropoutRate)
          dn.cmult(h, mask)
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

  def get_nll(x: Expression, labels: UnsignedVector, cg: ComputationGraph): Expression = {
    // compute output
    val y = run(x, cg)
    // do softmax
    val losses = dn.pickneglogsoftmax(y, labels)
    // sum across batches
    val result = dn.sum_batches(losses)
    result
  }

  def predict(x: Expression, cg: ComputationGraph): Int = {
    // run MLP to get class distribution
    val y = run(x, cg)
    // get values
    val probs = cg.forward(y).toSeq
    // return the argmax
    probs.zipWithIndex.max._2
  }

  def enable_dropout(): Unit = { dropoutActive = true }

  def disable_dropout(): Unit = { dropoutActive = false }

  def is_dropout_enabled(): Boolean = dropoutActive

  def activate(h: Expression, f: Activation.Value): Expression = f match {
    case Activation.LINEAR => h
    case Activation.RELU => dn.rectify(h)
    case Activation.SIGMOID => dn.logistic(h)
    case Activation.TANH => dn.tanh(h)
    case Activation.SOFTMAX => dn.softmax(h)
    case _ => throw new IllegalArgumentException("unknown activation function")
  }
}
