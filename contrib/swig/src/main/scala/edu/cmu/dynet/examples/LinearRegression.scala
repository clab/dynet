package edu.cmu.dynet.examples

import edu.cmu.dynet._

import scala.language.implicitConversions

object LinearRegression {
  def main(args: Array[String]) {
    Initialize.initialize()

    // x from 0.0 to 10.0
    val xs = (0 until 100).map(_.toFloat / 10)

    // y_i = 2 * x_i - 5 + epsilon
    val ys = for {
      x <- xs
      r = scala.util.Random.nextGaussian()
    } yield (-5.0 + 2.0 * x + 0.0033 * r).toFloat

    val model = new ParameterCollection
    val trainer = new SimpleSGDTrainer(model, 0.01f)
    ComputationGraph.renew()

    val p_W = model.addParameters(Dim(1))
    val W = Expression.parameter(p_W)

    val p_b = model.addParameters(Dim(1))
    val b = Expression.parameter(p_b)

    for (iter <- 1 to 20) {
      // track the total error for each iteration
      var iterLoss = 0f
      for ((x, y) <- xs.zip(ys)) {
        val prediction = W * x + b
        val loss = Expression.square(prediction - y)
        iterLoss += ComputationGraph.forward(loss).toFloat
        ComputationGraph.backward(loss)
        trainer.update()
      }

      // print the current parameter values
      val W_ = p_W.values.toFloat
      val b_ = p_b.values.toFloat
      println(s"(iter $iter) y = $W_ x + $b_ (error: $iterLoss)")
    }
  }
}
