package edu.cmu.dynet

import org.scalatest._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class ComputationGraphSpec extends FlatSpec with Matchers {

  import DynetScalaHelpers._

  "ComputationGraph" should "allow repeated calls to getNew" in {
    for (_ <- 1 to 100) {
      val cg = ComputationGraph.getNew
      cg.add_input(10)
    }
  }
}