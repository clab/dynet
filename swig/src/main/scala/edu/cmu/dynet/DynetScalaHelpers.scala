package src.main.scala.edu.cmu.dynet

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

import scala.language.implicitConversions

object DynetScalaHelpers {

  def dim(dims: Int*): Dim = {
    val dimInts = new LongVector
    dims.foreach(dimInts.add)
    new Dim(dimInts)
  }

  implicit class Untensor(t: Tensor) {
    def toFloat: Float = as_scalar(t)
    def toVector: FloatVector = as_vector(t)
  }

  implicit class RichExpression(e: Expression) {
    def +(e2: Expression): Expression = exprPlus(e, e2)
    def *(e2: Expression): Expression = exprTimes(e, e2)
    def -(e2: Expression): Expression = exprMinus(e, e2)
    def +(r: Float): Expression = exprPlus(e, r)
    def *(r: Float): Expression = exprTimes(e, r)
    def -(r: Float): Expression = exprMinus(e, r)
  }
}
