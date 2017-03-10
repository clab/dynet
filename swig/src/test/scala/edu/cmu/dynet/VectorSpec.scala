package edu.cmu.dynet

import org.scalatest._

class VectorSpec extends FlatSpec with Matchers {

  import Utilities._

  "FloatVector" should "construct when given a Seq[Float]" in {
    val fv = new FloatVector(Seq(2.3f, 4.5f))

    fv.size shouldBe 2
    fv(0) shouldBe 2.3f
    fv(1) shouldBe 4.5f
  }

  /*
  "FloatVector" should "construct when given a Seq[Double]" in {
    val fv = FloatVector(Seq(2.3, 4.5))

    fv.size shouldBe 2
    fv(0) shouldBe 2.3f
    fv(1) shouldBe 4.5f
  }
  */

  /*
  "DoubleVector" should "construct when given a Seq[Double]" in {
    val dv = DoubleVector(Seq(2.3, 4.5, -10.2))

    dv.size shouldBe 3
    dv.get(0) shouldBe 2.3
    dv.get(1) shouldBe 4.5
    dv.get(2) shouldBe -10.2
  }
  */

  "IntVector" should "construct when given a Seq[Int]" in {
    val iv = new IntVector(Seq(23, 45, -102))

    iv.size shouldBe 3
    iv(0) shouldBe 23
    iv(1) shouldBe 45
    iv(2) shouldBe -102
  }
}