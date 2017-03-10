package edu.cmu.dynet

import org.scalatest._

class ComputationGraphSpec extends FlatSpec with Matchers {

  "ComputationGraph" should "allow repeated calls to renew" in {
    for (_ <- 1 to 100) {
      ComputationGraph.renew()
      ComputationGraph.addInput(10)
    }
  }
}