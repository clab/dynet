package edu.cmu.dynet.examples

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import edu.cmu.dynet.DynetScalaHelpers

object XorScala {
  val HIDDEN_SIZE = 8
  val ITERATIONS = 30

  import DynetScalaHelpers._

  def main(args: Array[String]) {
    println("Running XOR example")
    myInitialize()
    println("Dynet initialized!")
    val m = new Model
    val sgd = new SimpleSGDTrainer(m)
    val cg = new ComputationGraph

    val p_W = m.add_parameters(dim(HIDDEN_SIZE, 2))
    val p_b = m.add_parameters(dim(HIDDEN_SIZE))
    val p_V = m.add_parameters(dim(1, HIDDEN_SIZE))
    val p_a = m.add_parameters(dim(1))

    val W = parameter(cg, p_W)
    val b = parameter(cg, p_b)
    val V = parameter(cg, p_V)
    val a = parameter(cg, p_a)

    val x_values = new FloatVector(2)
    val x = input(cg, dim(2), x_values)
    var y_value = 0f
    val y = input(cg, y_value)

    val h = tanh(W * x + b)
    val y_pred = V * h + a
    val loss_expr = squared_distance(y_pred, y)

    println()
    println("Computation graphviz structure:")
    cg.print_graphviz
    println()
    println("Training...")

    for (iter <- 0 to ITERATIONS - 1) {
      var loss: Float = 0
      for (mi <- 0 to 3) {
        val x1: Boolean = mi % 2 > 0
        val x2: Boolean = (mi / 2) % 2 > 0
        x_values.set(0, if (x1) 1 else -1)
        x_values.set(1, if (x2) 1 else -1)
        y_value = if (x1 != x2) 1 else -1
        loss += as_scalar(cg.forward(loss_expr))
        cg.backward(loss_expr)
        sgd.update(1.0f)
      }
      sgd.update_epoch
      loss /= 4
      println("iter = " + iter + ", loss = " + loss)
    }
  }
}
