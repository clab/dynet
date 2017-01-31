package edu.cmu.dynet

import edu.cmu.dynet.dynet_swig._

import scala.language.implicitConversions

object DynetScalaHelpers {

  import scala.collection.JavaConverters._
  import java.util.Collection

  // The collection constructors for the _Vector types require java.util.Collection[javatype] input,
  // so here are some implicit conversions from Seq[scalatype] to make them easier to work with
  implicit def convertFloatsToFloats(values: Seq[Float]): Collection[java.lang.Float] = {
    values.map(float2Float).asJavaCollection
  }

  implicit def convertDoublesToFloats(values: Seq[Double]): Collection[java.lang.Float] = {
    convertFloatsToFloats(values.map(_.toFloat))
  }

  implicit def convertDoublesToDoubles(values: Seq[Double]): Collection[java.lang.Double] = {
    values.map(double2Double).asJavaCollection
  }

  implicit def convertIntsToIntegers(values: Seq[Int]): Collection[java.lang.Integer] = {
    values.map(int2Integer).asJavaCollection
  }

  // The SWIG wrappers around pointers to C++ primitives are not very Scala-like to work with;
  // these are more Scala-y wrappers that implicitly convert to the SWIG versions.
  class FloatPointer {
    val floatp = new_floatp

    def set(value: Float): Unit = floatp_assign(floatp, value)

    def value(): Float = floatp_value(floatp)
  }

  implicit def toFloatp(fp: FloatPointer): SWIGTYPE_p_float = fp.floatp

  class IntPointer {
    val intp = new_intp

    def set(value: Int): Unit = intp_assign(intp, value)

    def value(): Int = intp_value(intp)
  }

  implicit def toIntp(ip: IntPointer): SWIGTYPE_p_int = ip.intp

  def dim(dims: Int*): Dim = {
    val dimInts = new LongVector
    dims.foreach(dimInts.add)
    new Dim(dimInts)
  }

  implicit def seqToDim(dims: Seq[Int]): Dim = {
    new Dim(dims.map(_.toLong):_*)
  }

  implicit class Untensor(t: Tensor) {
    def toFloat: Float = as_scalar(t)
    def toVector: FloatVector = as_vector(t)
    def toSeq: Seq[Float] = {
      val vector = t.toVector
      for (i <- 0 until vector.size.toInt) yield vector.get(i)
    }
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
