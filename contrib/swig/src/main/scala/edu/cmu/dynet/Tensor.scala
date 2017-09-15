package edu.cmu.dynet

/** DyNet operates on `Tensor` objects. You should never need to construct them directly;
  * however, here are some methods for getting values out of them.
  */
class Tensor private[dynet] (private[dynet] val tensor: internal.Tensor) {
  def toFloat(): Float = internal.dynet_swig.as_scalar(tensor)
  def toVector(): FloatVector = new FloatVector(internal.dynet_swig.as_vector(tensor))
  def toSeq(): Seq[Float] = toVector()
  def getD(): Dim = new Dim(tensor.getD)
}

