package edu.cmu.dynet

import org.scalatest._
import Matchers._

case class RegressionLine(slope: Float, intercept: Float) {}

/* An end-to-end example for sanity check purposes. */
class LinearRegressionSpec extends FlatSpec with Matchers {

  import Utilities._

  Initialize.initialize()

  def regress(xs: Seq[Float], ys: Seq[Float], numIterations: Int = 20): RegressionLine = {
    assert(xs.size > 0)
    assert(xs.size == ys.size)

    val model = new ParameterCollection
    val trainer = new SimpleSGDTrainer(model, 0.01f)

    val p_W = model.addParameters(Dim(1))
    val p_b = model.addParameters(Dim(1))

    val examples = xs.zip(ys)

    for (iter <- 1 to numIterations) {
      for ((x, y) <- examples) {
        ComputationGraph.renew()
        val W = Expression.parameter(p_W)
        val b = Expression.parameter(p_b)

        val prediction = W * x + b
        val loss = Expression.square(prediction - y)
        ComputationGraph.forward(loss)
        ComputationGraph.backward(loss)
        trainer.update()
      }
    }

    RegressionLine(p_W.values.toFloat, p_b.values.toFloat)
  }

  "regression" should "learn the correct model" in {
    // x from 0.0 to 10.0
    val xs = (0 until 100).map(_.toFloat / 10)

    // y_i = 2 * x_i
    val ys = for {
      x <- xs
    } yield (-5.0 + 2.0 * x).toFloat

    val result = regress(xs, ys, 20)

    // These are very weak bounds, 20 iterations should always get this close.
    result.slope shouldBe 2f +- 1f
    result.intercept shouldBe -5f +- 1f
  }
}
