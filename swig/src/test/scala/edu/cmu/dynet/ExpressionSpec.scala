package edu.cmu.dynet

import org.scalatest._
import Matchers._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class ExpressionSpec extends FlatSpec with Matchers {

  import DynetScalaHelpers._
  myInitialize()

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

  "simple expression functions" should "do the right things" in {
    val cg = ComputationGraph.getNew

    val e1 = input(cg, 1)
    val e2 = input(cg, 2)
    val e3 = input(cg, 3)

    // arithmetic
    exprMinus(e1) shouldHaveValue -1f
    exprPlus(e1, e2) shouldHaveValue 3f
    exprPlus(e1, 10) shouldHaveValue 11f
    exprPlus(10, e1) shouldHaveValue 11f
    exprMinus(e2, e1) shouldHaveValue 1f
    exprMinus(e2, 10) shouldHaveValue -8f
    exprMinus(10, e2) shouldHaveValue 8f
    exprTimes(e1, e2) shouldHaveValue 2f
    exprTimes(10, e2) shouldHaveValue 20f
    exprTimes(e2, 10) shouldHaveValue 20f
    exprDivide(e2, 10) shouldHaveValue 0.2f

    // affine transform
    affine_transform(Seq(e1, e2, e3)) shouldHaveValue 7 // 1 + 2 * 3
    affine_transform(Seq(e1, e2, e3, e1, e3)) shouldHaveValue 10 // 1 + 2 * 3 + 1 * 3

    // sum + average
    sum(Seq(e1, e2, e2, e3, e3, e3)) shouldHaveValue 14
    average(Seq(e1, e2, e2, e3, e3, e3)) shouldHaveValue 14f / 6

    val sqrt2 = sqrt(e2)
    exprTimes(sqrt2, sqrt2) shouldHaveValue 2

    square(e3) shouldHaveValue 9
    pow(e2, e3) shouldHaveValue 8
    pow(e3, e2) shouldHaveValue 9

    min(e1, e3) shouldHaveValue 1
    max(e1, e3) shouldHaveValue 3

    // TODO(joelgrus): write more tests
  }
}