package edu.cmu.dynet

import org.scalatest._
import edu.cmu.dynet.internal.{dynet_swig => dn}

class ParameterInitSpec extends FlatSpec with Matchers {

  Initialize.initialize()

  "ParameterInitConst" should "set constant values" in {
    val model = new ParameterCollection
    val p_W = model.addParameters(Dim(10))

    val init = ParameterInit.const(10.0f)
    init.initializeParams(p_W.values)

    // All values should be initialized to 10.0f
    p_W.values.toSeq.foreach(x => x shouldBe 10.0f)
  }

  "ParameterInitNormal" should "initialize things normally" in {
    val model = new ParameterCollection
    val p_W = model.addParameters(Dim(10000))

    // mean 10, variance 4
    val init = ParameterInit.normal(10, 4)
    init.initializeParams(p_W.values)

    val values = p_W.values.toSeq

    // Incredibly weak bounds on the sample mean and variance, basically just a sanity check.
    val mean = values.sum / 10000
    mean > 7  shouldBe true
    mean < 13 shouldBe true

    val s2 = values.map(v => scala.math.pow(v - mean, 2)).sum / 9999f
    s2 > 2 shouldBe true
    s2 < 6 shouldBe true
  }

  "ParameterInitUniform" should "initialize things uniformly" in {
    val model = new ParameterCollection
    val p_W = model.addParameters(Dim(10000))

    // uniform from 12 to 17
    val init = ParameterInit.uniform(12f, 17f)
    init.initializeParams(p_W.values)

    val values = p_W.values.toSeq

    values.max <= 17 shouldBe true
    values.min >= 12 shouldBe true

    // Incredibly weak bounds on the sample mean and variance, basically just a sanity check.
    val mean = values.sum / 10000
    mean > 13 shouldBe true
    mean < 16 shouldBe true

    // Must take on values in the top / bottom quartile
    values.exists(v => v > 15.75) shouldBe true
    values.exists(v => v < 13.25) shouldBe true
  }

  "ParameterInitIdentity" should "initialize to the identity matrix" in {
    val model = new ParameterCollection
    val p_W = model.addParameters(Dim(100, 100))

    val init = ParameterInit.identity()

    init.initializeParams(p_W.values)
    val values = p_W.values.toSeq

    for {
      i <- 0 until 100
      j <- 0 until 100
      z = if (i == j) 1f else 0f
    } {
      values(i * 100 + j) shouldBe z
    }
  }

  "ParameterInitFromVector" should "initialize from a vector" in {
    val model = new ParameterCollection
    val p_W = model.addParameters(Dim(1000))

    val valuesIn = (1 to 1000).map(x => math.sin(x).toFloat)
    val vector = new FloatVector(valuesIn)
    val init = ParameterInit.fromVector(vector)

    init.initializeParams(p_W.values)

    val valuesOut = p_W.values.toSeq

    valuesIn.zip(valuesOut).foreach {
      case (vi, vo) => vi shouldBe vo
    }
  }

  "ParameterInitGlorot" should "initialize using the correct distribution" in {
    val model = new ParameterCollection
    val p_W = model.addParameters(Dim(20, 5))

    val init = ParameterInit.glorot()

    init.initializeParams(p_W.values)

    // should be uniform [-s, s] where s = sqrt(6) / sqrt(25) = sqrt(6) / 5
    val s = math.sqrt(6) / 5 // ~ 0.4899

    val values = p_W.values.toSeq

    // within the bounds
    values.min > -s shouldBe true
    values.max <  s shouldBe true

    // reasonable looking mean
    val mean = values.sum / 100
    mean <  0.3 shouldBe true
    mean > -0.3 shouldBe true

    // but some dispersion
    values.exists(v => v > s  / 4) shouldBe true
    values.exists(v => v < -s / 4) shouldBe true
  }
}
