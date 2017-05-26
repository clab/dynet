package edu.cmu.dynet

/** Represents a "dimension", which you should think of as the dimension of a tensor. Can only be
  *  constructed using factory methods in the companion object.
  *
  * @param dim
  */
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

  /** We override `equals` so that `Dim` objects should be equal whenever all of their dimension
    * sizes match.
    */
  override def equals(that: Any) = that match {
    case that: Dim => dim == that.dim
    case _ => false
  }
  override def hashCode(): Int = dim.hashCode()

  override def toString: String = "Dim(" + (0 until nDims.toInt).map(get(_)).mkString(", ") + ")"

  def debugString(): String = s"(Dim: ${size} ${nDims} ${(0 until nDims.toInt).map(get(_))} )"
}

/** Factory for [[edu.cmu.dynet.Dim]] instances. */
object Dim {
  /** Creates a Dim object from a `Seq` of dimensions and a batch size
    *
    * @param values a `Seq` of dimensions
    * @param b the batch size (zero by default)
    */
  def apply(values: Seq[Int], b: Long = 0): Dim = {
    val lv = new internal.LongVector()
    values.foreach(lv.add)
    val dim: internal.Dim = if (b > 0) new internal.Dim(lv, b) else new internal.Dim(lv)
    new Dim(dim)
  }

  /** Creates a Dim object from a list of the dimensions
    *
    * @param values a list of the dimensions
    */
  def apply(values: Int*): Dim = apply(values)
}

