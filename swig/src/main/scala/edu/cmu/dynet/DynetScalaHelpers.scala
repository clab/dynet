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

  implicit def convertExpressionsToExpressions(values: Seq[Expression]): Collection[Expression] = {
    values.asJavaCollection
  }

  // shuffle indices
  def shuffle(vs: IntVector): Unit = {
    val values = for (i <- 0 until vs.size.toInt) yield vs.get(i)
    scala.util.Random.shuffle(values).zipWithIndex.foreach { case (v, i) => vs.set(i, v) }
  }
  
  // sample from a discrete distribution
  def sample(v: FloatVector): Int = {
    // random pick
    val p = scala.util.Random.nextFloat

    // Seq(0f, p(0), p(0) + p(1), .... )
    val cumulative = v.scanLeft(0f)(_ + _)

    // Return the largest index where the cumulative probability is <= p.
    // Since cumulative(0) is 0f, there's always at least one element in the
    // takeWhile, so it's ok to use .last
    cumulative.zipWithIndex
        .takeWhile { case (c, i) => c <= p }
        .last
        ._2
  }

  // Convert vectors to Seqs for easy iteration.
  implicit def floatVectorToSeq(fv: FloatVector): Seq[Float] = {
    for (i <- 0 until fv.size.toInt) yield fv.get(i)
  }

  implicit def intVectorToSeq(iv: IntVector): Seq[Int] = {
    for (i <- 0 until iv.size.toInt) yield iv.get(i)
  }

  implicit def unsignedVectorToSeq(uv: UnsignedVector): Seq[Long] = {
    for (i <- 0 until uv.size.toInt) yield uv.get(i)
  }

  implicit def expressionVectorToSeq(ev: ExpressionVector): Seq[Expression] = {
    for (i <- 0 until ev.size.toInt) yield ev.get(i)
  }

  implicit def parameterVectorToSeq(pv: ParameterStorageVector): Seq[ParameterStorage] = {
    for (i <- 0 until pv.size.toInt) yield pv.get(i)
  }

  implicit def lookupParameterVectorToSeq(
    pv: LookupParameterStorageVector
  ): Seq[LookupParameterStorage] = {
    for (i <- 0 until pv.size.toInt) yield pv.get(i)
  }

  // The SWIG wrappers around pointers to C++ primitives are not very Scala-like to work with;
  // these are more Scala-y wrappers that implicitly convert to the SWIG versions.
  class FloatPointer {
    val floatp = new_floatp
    set(0f)

    def set(value: Float): Unit = floatp_assign(floatp, value)

    def value(): Float = floatp_value(floatp)
  }

  implicit def toFloatp(fp: FloatPointer): SWIGTYPE_p_float = fp.floatp

  class IntPointer {
    val intp = new_intp
    set(0)

    def set(value: Int): Unit = intp_assign(intp, value)

    def value(): Int = intp_value(intp)

    def increment(by: Int = 1) = set(value + by)
  }

  implicit def toIntp(ip: IntPointer): SWIGTYPE_p_int = ip.intp

  class UnsignedPointer {
    val uintp = new_uintp()
    set(0)

    def set(value: Int): Unit = uintp_assign(uintp, value)

    def value(): Int = uintp_value(uintp).toInt
  }

  implicit def toUnsignedp(up: UnsignedPointer): SWIGTYPE_p_unsigned_int = up.uintp

  // This is helpful for debugging.
  def show(dim: Dim, prefix: String=""): Unit = {
    val dims = for (i <- 0 until dim.ndims().toInt) yield dim.get(i)
    val dimstring = dims.mkString(",")
    val bd = if (dim.batch_elems != 1) s"X${dim.batch_elems}" else ""
    println(s"$prefix{$dimstring$bd}")
  }

  def dim(dims: Int*): Dim = {
    val dimInts = new LongVector
    dims.foreach(dimInts.add)
    new Dim(dimInts)
  }

  def dim(xs: Seq[Int], b: Int): Dim = {
    val lv = new LongVector
    xs.foreach(lv.add)
    new Dim(lv, b)
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

  type NamedParameters = Map[String, Parameter]
  type NamedLookupParameters = Map[String, LookupParameter]

  implicit class ExtraSavers(saver: ModelSaver) {

    def add_string(s: String): Unit = saver.add_object(s)

    def add_named_parameters(np: NamedParameters): Unit = {
      saver.add_int(np.size)
      for ((name, parameter) <- np) {
        saver.add_string(name)
        saver.add_parameter(parameter)
      }
    }

    def add_named_lookup_parameters(np: NamedLookupParameters): Unit = {
      saver.add_int(np.size)
      for ((name, parameter) <- np) {
        saver.add_string(name)
        saver.add_lookup_parameter(parameter)
      }
    }
  }

  implicit class ExtraLoaders(loader: ModelLoader) {

    def load_string(): String = loader.load_object(classOf[String])

    def load_named_parameters(): NamedParameters = {
      val numParams = loader.load_int()
      (for {
        i <- 1 to numParams
        name = loader.load_string()
        parameter = loader.load_parameter()
      } yield (name, parameter)).toMap
    }

    def load_named_lookup_parameters(): NamedLookupParameters = {
      val numParams = loader.load_int()
      (for {
        i <- 1 to numParams
        name = loader.load_string()
        parameter = loader.load_lookup_parameter()
      } yield (name, parameter)).toMap
    }
  }

  // Sugar to turn `Expression` operators into Scala operators.
  implicit class RichExpression(e: Expression) {
    def +(e2: Expression): Expression = exprPlus(e, e2)
    def *(e2: Expression): Expression = exprTimes(e, e2)
    def -(e2: Expression): Expression = exprMinus(e, e2)
    def +(r: Float): Expression = exprPlus(e, r)
    def *(r: Float): Expression = exprTimes(e, r)
    def -(r: Float): Expression = exprMinus(e, r)
    def /(r: Float): Expression = exprDivide(e, r)
    def unary_-(): Expression = exprMinus(e)
  }

  implicit class RichNumeric[T](x: T)(implicit n: Numeric[T]) {
    import n._
    def +(e: Expression): Expression = exprPlus(x.toFloat, e)
    def *(e: Expression): Expression = exprTimes(x.toFloat, e)
    def -(e: Expression): Expression = exprMinus(x.toFloat, e)
  }
}
