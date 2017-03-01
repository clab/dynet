package edu.cmu.dynet

import org.scalatest._

class DimSpec extends FlatSpec with Matchers {

  import DyNetScalaHelpers._

  "Dims" should "get constructed properly with the varargs constructor" in {
    val dim123 = new Dim(1, 2, 3)

    dim123.ndims() shouldBe 3
    dim123.get(0) shouldBe 1
    dim123.get(1) shouldBe 2
    dim123.get(2) shouldBe 3
  }

  "Dims" should "get constructed properly with the helper function" in {
    val dim123 = dim(1, 2, 3)

    dim123.ndims() shouldBe 3
    dim123.get(0) shouldBe 1
    dim123.get(1) shouldBe 2
    dim123.get(2) shouldBe 3
  }

  "Dims" should "get implicitly constructed" in {
    val dim123: Dim = Seq(1, 2, 3)

    dim123.ndims() shouldBe 3
    dim123.get(0) shouldBe 1
    dim123.get(1) shouldBe 2
    dim123.get(2) shouldBe 3
  }

  "Dims" should "be equal when they're the same" in {
    val dim123 = new Dim(1, 2, 3)
    val dim12 = new Dim(1, 2)
    val dim123_ = new Dim(1, 2, 3)
    val dim1234 = new Dim(1, 2, 3, 4)
    val dim124 = new Dim(1, 2, 4)

    dim123  == dim123_ shouldBe true
    dim123  == dim12   shouldBe false
    dim123_ == dim12   shouldBe false
    dim123 == dim1234  shouldBe false
    dim123 == dim124   shouldBe false

    dim123.hashCode()  == dim123_.hashCode() shouldBe true
    dim123.hashCode()  == dim12.hashCode()   shouldBe false
    dim123_.hashCode() == dim12.hashCode()   shouldBe false
  }
}