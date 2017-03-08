package edu.cmu.dynet

import scala.language.implicitConversions
import scala.collection.JavaConverters._
import ImplicitConversions._

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

  override def apply(idx: Int): Expression = new Expression(vector.get(idx))
  override def length: Int = vector.size.toInt
  override def update(idx: Int, elem: Expression): Unit = {
    elem.ensureFresh()
    vector.set(idx, elem)
  }
}