package edu.cmu.dynet

/** Behind the scenes, DyNet frequently operates on C++ `std::vector<>` types. The wrapper
  * classes implement [[scala.collection.mutable.IndexedSeq]] to make them easy to work with
  * in Scala code. Each has a `size: Long` constructor and a `values: Seq[_]` constructor.
  */

import scala.language.implicitConversions
import scala.collection.JavaConverters._

object IntVector {
  implicit def Seq2IntVector(x: Seq[Int]) =
    new IntVector(x)
}

class IntVector private[dynet] (private[dynet] val vector: internal.IntVector)
    extends scala.collection.mutable.IndexedSeq[Int] {
  def this(size: Long) { this(new internal.IntVector(size)) }
  def this(values: Seq[Int] = Seq.empty) {
    this(new internal.IntVector(values.map(int2Integer).asJavaCollection))
  }

  def add(v: Int): Unit = vector.add(v)

  override def apply(idx: Int): Int = vector.get(idx)
  override def length: Int = vector.size.toInt
  override def update(idx: Int, elem: Int): Unit = vector.set(idx, elem)
}

object UnsignedVector {
  implicit def Seq2UnsignedVector(x: Seq[Long]) =
    new UnsignedVector(x)
}

/** SWIG converts C++ `unsigned` to Scala `Long` */
class UnsignedVector private[dynet] (private[dynet] val vector: internal.UnsignedVector)
    extends scala.collection.mutable.IndexedSeq[Long] {
  def this(size: Long) { this(new internal.UnsignedVector(size)) }
  def this(values: Seq[Long] = Seq.empty) {
    this(new internal.UnsignedVector(values.map(_.toInt).map(int2Integer).asJavaCollection))
  }

  def add(v: Long): Unit = vector.add(v)
  override def apply(idx: Int): Long = vector.get(idx)
  override def length: Int = vector.size.toInt
  override def update(idx: Int, elem: Long): Unit = vector.set(idx, elem)
}

object FloatVector {
  implicit def Seq2FloatVector(x: Seq[Float]) =
    new FloatVector(x)
}

class FloatVector private[dynet] (private[dynet] val vector: internal.FloatVector)
    extends scala.collection.mutable.IndexedSeq[Float] {
  def this(size: Long) { this(new internal.FloatVector(size)) }
  def this(values: Seq[Float] = Seq.empty) {
    this(new internal.FloatVector(values.map(float2Float).asJavaCollection))
  }

  def add(v: Float): Unit = vector.add(v)
  override def apply(idx: Int): Float = vector.get(idx)
  override def length: Int = vector.size.toInt
  override def update(idx: Int, elem: Float): Unit = vector.set(idx, elem)
}

object ExpressionVector {
  implicit def Seq2ExpressionVector(x: Seq[Expression]) =
    new ExpressionVector(x)
}

class ExpressionVector private[dynet] (
  private[dynet] val version: Long, private[dynet] val vector: internal.ExpressionVector)
    extends scala.collection.mutable.IndexedSeq[Expression] {
  private[dynet] def this(vector: internal.ExpressionVector) = {
    this(ComputationGraph.version, vector)
  }

  def this(size: Long) { this(new internal.ExpressionVector(size)) }
  def this(values: Seq[Expression] = Seq.empty) {
    this(new internal.ExpressionVector(values.map(_.expr).asJavaCollection))
    ensureFresh()
  }

  def ensureFresh(): Unit = {
    if (version != ComputationGraph.version) {
      throw new RuntimeException("stale")
    }
  }

  def add(v: Expression): Unit = {
    v.ensureFresh()
    vector.add(v.expr)
  }

  // The vector must be kept around in order to index into it.
  override def apply(idx: Int): Expression = new Expression(vector.get(idx), this)
  override def length: Int = vector.size.toInt
  override def update(idx: Int, elem: Expression): Unit = {
    elem.ensureFresh()
    vector.set(idx, elem.expr)
  }
}

object UnsignedVectorVector {
  implicit def Seq2UnsignedVectorVector(x: Seq[UnsignedVector]) =
    new UnsignedVectorVector(x)
}

class UnsignedVectorVector private[dynet] (private[dynet] val vector: internal.UnsignedVectorVector)
  extends scala.collection.mutable.IndexedSeq[UnsignedVector] {
  def this(size: Long) { this(new internal.UnsignedVectorVector(size)) }
  def this(values: Seq[UnsignedVector] = Seq.empty) {
    this(new internal.UnsignedVectorVector(values.map(_.vector).asJavaCollection))
  }

  def add(v: UnsignedVector): Unit = vector.add(v.vector)
  override def apply(idx: Int): UnsignedVector = new UnsignedVector(vector.get(idx))
  override def length: Int = vector.size.toInt
  override def update(idx: Int, v: UnsignedVector): Unit = vector.set(idx, v.vector)
}
