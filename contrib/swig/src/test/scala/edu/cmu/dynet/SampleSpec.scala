package edu.cmu.dynet

import org.scalatest._

class SampleSpec extends FlatSpec with Matchers {

  import Utilities._

  "Sample" should "do the right thing" in {

    val probs = new FloatVector(Seq(0.1f, 0.1f, 0.7f, 0.1f))

    val samples = for (_ <- 1 to 10000) yield sample(probs)

    // should take on all values
    samples.distinct.toSet shouldBe Set(0, 1, 2, 3)

    // at least half should be in the 70% class
    samples.filter(_ == 2).size > 5000 shouldBe true
  }
}