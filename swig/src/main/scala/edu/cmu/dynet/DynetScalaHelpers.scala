package edu.cmu.dynet

import edu.cmu.dynet.dynet_swig._

import scala.language.implicitConversions

object DynetScalaHelpers {

  // The SWIG wrappers around pointers to C++ primitives are not very Scala-like to work with,
  // these are more Scala-y wrappers that implicitly converts to the SWIG version.
  class FloatPointer {
    val floatp = new_floatp

    def set(value: Float) = floatp_assign(floatp, value)

    def value() = floatp_value(floatp)
  }

  implicit def toFloatp(fp: FloatPointer): SWIGTYPE_p_float = fp.floatp

  class IntPointer {
    val intp = new_intp

    def set(value: Int) = intp_assign(intp, value)

    def value() = intp_value(intp)
  }

  implicit def toIntp(ip: IntPointer): SWIGTYPE_p_int = ip.intp

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
