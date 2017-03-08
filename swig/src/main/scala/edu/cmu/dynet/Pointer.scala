package edu.cmu.dynet

import internal.{dynet_swig => dn}

// The SWIG wrappers around pointers to C++ primitives are not very Scala-like to work with;
// these are more Scala-y wrappers that implicitly convert to the SWIG versions.
class FloatPointer {
  val floatp = dn.new_floatp
  set(0f)

  def set(value: Float): Unit = dn.floatp_assign(floatp, value)

  def value(): Float = dn.floatp_value(floatp)
}

class IntPointer {
  val intp = dn.new_intp
  set(0)

  def set(value: Int): Unit = dn.intp_assign(intp, value)

  def value(): Int = dn.intp_value(intp)

  def increment(by: Int = 1) = set(value + by)
}

class UnsignedPointer {
  val uintp = dn.new_uintp()
  set(0)

  def set(value: Int): Unit = dn.uintp_assign(uintp, value)

  def value(): Int = dn.uintp_value(uintp).toInt
}