package edu.cmu.dynet

import scala.language.implicitConversions

object DyNetScalaHelpers {

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
    val values = for (i <- 0 until vs.size) yield vs(i)
    scala.util.Random.shuffle(values).zipWithIndex.foreach { case (v, i) => vs.update(i, v) }
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



  /*
  // This is helpful for debugging.
  def show(dim: Dim, prefix: String=""): Unit = {
    val dims = for (i <- 0 until dim.ndims().toInt) yield dim.get(i)
    val dimstring = dims.mkString(",")
    val bd = if (dim.batch_elems != 1) s"X${dim.batch_elems}" else ""
    println(s"$prefix{$dimstring$bd}")
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
  */

  implicit class RichNumeric[T](x: T)(implicit n: Numeric[T]) {
    import n._
    def +(e: Expression): Expression = Expression.exprPlus(x.toFloat, e)
    def *(e: Expression): Expression = Expression.exprTimes(x.toFloat, e)
    def -(e: Expression): Expression = Expression.exprMinus(x.toFloat, e)
  }
}
