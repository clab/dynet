package edu.cmu.dynet

class Tensor private[dynet] (private[dynet] val tensor: internal.Tensor) {
  def toFloat(): Float = internal.dynet_swig.as_scalar(tensor)
  def toVector(): FloatVector = new FloatVector(internal.dynet_swig.as_vector(tensor))
  def toSeq(): Seq[Float] = toVector()
}

