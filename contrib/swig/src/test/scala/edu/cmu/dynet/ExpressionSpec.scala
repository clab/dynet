package edu.cmu.dynet

import org.scalatest._
import Matchers._

class ExpressionSpec extends FlatSpec with Matchers {

  import Utilities._
  Initialize.initialize()

  implicit def expressionV(es: Seq[Expression]): ExpressionVector = {
    new ExpressionVector(es)
  }

  // implicit class for pulling values out of Expressions and testing them
  implicit class ShouldExpressionBe(expression: Expression) {
    val episilon = 1e-4f

    def shouldHaveValue(value: Float): Unit = {
      expression.value.toFloat shouldBe (value +- episilon)
    }
    
    def shouldHaveValues(expected: Seq[Float]): Unit = {
      val actuals = expression.value.toSeq
      expected.size shouldBe actuals.size
      expected.zip(actuals).foreach {
        case (e, a) => a shouldBe (e +- episilon)
      }
    }
  }

  "expression functions" should "do the right things" in {
    ComputationGraph.renew()

    val e1 = Expression.input(1)
    val e2 = Expression.input(2)
    val e3 = Expression.input(3)

    // arithmetic
    -e1 shouldHaveValue -1f
    e1 + e2 shouldHaveValue 3f
    e1 + 10 shouldHaveValue 11f
    10 + e1 shouldHaveValue 11f
    e2 - e1 shouldHaveValue 1f
    e2 - 10 shouldHaveValue -8f
    10 - e2 shouldHaveValue 8f
    e1 * e2 shouldHaveValue 2f
    10 * e2 shouldHaveValue 20f
    e2 * 10 shouldHaveValue 20f
    e2 / 10 shouldHaveValue 0.2f


    // affine transform
    Expression.affineTransform(Seq(e1, e2, e3)) shouldHaveValue 7 // 1 + 2 * 3
    Expression.affineTransform(Seq(e1, e2, e3, e1, e3)) shouldHaveValue 10 // 1 + 2 * 3 + 1 * 3

    // sum + average
    Expression.sum(Seq(e1, e2, e2, e3, e3, e3)) shouldHaveValue 14
    Expression.average(Seq(e1, e2, e2, e3, e3, e3)) shouldHaveValue 14f / 6

    val sqrt2 = Expression.sqrt(e2)
    Expression.exprTimes(sqrt2, sqrt2) shouldHaveValue 2

    Expression.square(e3) shouldHaveValue 9
    Expression.pow(e2, e3) shouldHaveValue 8
    Expression.pow(e3, e2) shouldHaveValue 9

    Expression.min(e1, e3) shouldHaveValue 1
    Expression.max(e1, e3) shouldHaveValue 3

    // TODO(joelgrus): write more tests
  }

  it should "fail gracefully" in {
    ComputationGraph.renew()
    assertThrows[AssertionError] {
      val foo = Expression.concatenate()
    }
  }

  "lists of expressions" should "get converted to vectors" in {
    ComputationGraph.renew()

    val exprs = for (i <- 1 to 100) yield Expression.input(i)

    val sums = for (i <- 1 to 50) yield Expression.sum(exprs: _*)
    sums.foreach(_ shouldHaveValue (1 to 100).sum)

    val uberSum = Expression.sum(
      new ExpressionVector(
        for {
          _ <- 1 to 1000
          i1 = scala.util.Random.nextInt(100)
          i2 = scala.util.Random.nextInt(100)
          i3 = scala.util.Random.nextInt(100)
        } yield Expression.sum(exprs(i1), exprs(i2), exprs(i3))
      )
    )

    val value = uberSum.value.toFloat
    value should be > 30f * 1000 * 3
    value should be < 70f * 1000 * 3
  }
}