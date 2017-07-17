package edu.cmu.dynet

import org.scalatest._

class DimSpec extends FlatSpec with Matchers {

  import Utilities._

  "Dims" should "get constructed properly with the varargs constructor" in {
    val dim123 = Dim(1, 2, 3)

    dim123.nDims() shouldBe 3
    dim123.get(0) shouldBe 1
    dim123.get(1) shouldBe 2
    dim123.get(2) shouldBe 3
  }

  "Dims" should "be equal when they're the same" in {
    val dim123 = Dim(1, 2, 3)
    val dim12 = Dim(1, 2)
    val dim123_ = Dim(1, 2, 3)
    val dim1234 = Dim(1, 2, 3, 4)
    val dim124 = Dim(1, 2, 4)

    dim123.dim == dim123_.dim shouldBe true
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