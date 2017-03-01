package edu.cmu.dynet

import org.scalatest._
import Matchers._
import edu.cmu.dynet.{dynet_swig => dn}

class ExpressionSpec extends FlatSpec with Matchers {

  import DyNetScalaHelpers._
  dn.initialize(new DynetParams)

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

    val e1 = dn.input(cg, 1)
    val e2 = dn.input(cg, 2)
    val e3 = dn.input(cg, 3)

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
    dn.affine_transform(Seq(e1, e2, e3)) shouldHaveValue 7 // 1 + 2 * 3
    dn.affine_transform(Seq(e1, e2, e3, e1, e3)) shouldHaveValue 10 // 1 + 2 * 3 + 1 * 3

    // sum + average
    dn.sum(Seq(e1, e2, e2, e3, e3, e3)) shouldHaveValue 14
    dn.average(Seq(e1, e2, e2, e3, e3, e3)) shouldHaveValue 14f / 6

    val sqrt2 = dn.sqrt(e2)
    dn.exprTimes(sqrt2, sqrt2) shouldHaveValue 2

    dn.square(e3) shouldHaveValue 9
    dn.pow(e2, e3) shouldHaveValue 8
    dn.pow(e3, e2) shouldHaveValue 9

    dn.min(e1, e3) shouldHaveValue 1
    dn.max(e1, e3) shouldHaveValue 3


    // TODO(joelgrus): write more tests
  }
}