package edu.cmu.dynet

class Dim private[dynet] (private[dynet] val dim: internal.Dim) {

  def size(): Long = dim.size
  def batchSize(): Long = dim.batch_size()
  def sumDims(): Long = dim.sum_dims()

  def truncate(): Dim = new Dim(dim.truncate())
  def singleBatch(): Dim = new Dim(dim.single_batch())

  def resize(i: Long) = dim.resize(i)
  def nDims(): Long = dim.ndims()
  def rows(): Long = dim.rows()
  def cols(): Long = dim.cols()
  def batchElems(): Long = dim.batch_elems()

  def set(i: Long, s: Long): Unit = dim.set(i, s)
  def get(i: Long): Long = dim.get(i)
  def size(i: Long): Long = dim.size(i)

  def deleteDim(i: Long): Unit = dim.delete_dim(i)

  def transpose(): Dim = new Dim(dim.transpose())

  override def equals(that: Any) = that match {
    case that: Dim => dim == that.dim
    case _ => false
  }
  override def hashCode(): Int = dim.hashCode()
}

// Dim has no public constructors, has to be constructed via factory methods.
object Dim {
  def apply(values: Seq[Int], b: Long = 0): Dim = {
    val lv = new internal.LongVector()
    values.foreach(lv.add)
    val dim: internal.Dim = if (b > 0) new internal.Dim(lv, b) else new internal.Dim(lv)
    new Dim(dim)
  }

  def apply(values: Int*): Dim = apply(values)
}

