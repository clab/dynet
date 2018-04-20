package edu.cmu.dynet.examples

import edu.cmu.dynet._

object XorScala {
  val HIDDEN_SIZE = 8
  val ITERATIONS = 30

  def main(args: Array[String]) {
    println("Running XOR example")
    Initialize.initialize()
    println("Dynet initialized!")
    val m = new ParameterCollection
    val sgd = new SimpleSGDTrainer(m)
    ComputationGraph.renew()

    val p_W = m.addParameters(Dim(HIDDEN_SIZE, 2))
    val p_b = m.addParameters(Dim(HIDDEN_SIZE))
    val p_V = m.addParameters(Dim(1, HIDDEN_SIZE))
    val p_a = m.addParameters(Dim(1))

    val W = Expression.parameter(p_W)
    val b = Expression.parameter(p_b)
    val V = Expression.parameter(p_V)
    val a = Expression.parameter(p_a)

    val x_values = new FloatVector(2)
    val x = Expression.input(Dim(2), x_values)

    // Need a pointer representation of scalars so updates are tracked
    val y_value = new FloatPointer
    y_value.set(0)
    val y = Expression.input(y_value)

    val h = Expression.tanh(W * x + b)
    val y_pred = V * h + a
    val loss_expr = Expression.squaredDistance(y_pred, y)

    println()
    println("Computation graphviz structure:")
    ComputationGraph.printGraphViz()
    println()
    println("Training...")

    for (iter <- 0 to ITERATIONS - 1) {
      var loss: Float = 0
      for (mi <- 0 to 3) {
        val x1: Boolean = mi % 2 > 0
        val x2: Boolean = (mi / 2) % 2 > 0
        x_values.update(0, if (x1) 1 else -1)
        x_values.update(1, if (x2) 1 else -1)
        y_value.set(if (x1 != x2) 1 else -1)
        loss += ComputationGraph.forward(loss_expr).toFloat
        ComputationGraph.backward(loss_expr)
        sgd.update()
      }
      sgd.learningRate *= 0.998f
      loss /= 4
      println("iter = " + iter + ", loss = " + loss)
    }
  }
}
