package edu.cmu.dynet

import internal.{dynet_swig => dn}

/** For the most part, SWIG makes it so you just use Scala classes where you would use C++ pointers
  * to class instances. The exception is for pointers to primitives. The SWIG wrappers for those are
  * not Scala-like at all, so here are wrapper that are easier to work with.
  */
class FloatPointer {
  val floatp = dn.new_floatp
  set(0f)

  def set(value: Float): Unit = dn.floatp_assign(floatp, value)

  def value(): Float = dn.floatp_value(floatp)

  override protected def finalize(): Unit = {
    dn.delete_floatp(floatp)
  }
}

class IntPointer {
  val intp = dn.new_intp
  set(0)

  def set(value: Int): Unit = dn.intp_assign(intp, value)

  def value(): Int = dn.intp_value(intp)

  def increment(by: Int = 1) = set(value + by)

  override protected def finalize(): Unit = {
    dn.delete_intp(intp)
  }
}

class UnsignedPointer {
  val uintp = dn.new_uintp()
  set(0)

  def set(value: Int): Unit = dn.uintp_assign(uintp, value)

  def value(): Int = dn.uintp_value(uintp).toInt

  override protected def finalize(): Unit = {
    dn.delete_uintp(uintp)
  }
}