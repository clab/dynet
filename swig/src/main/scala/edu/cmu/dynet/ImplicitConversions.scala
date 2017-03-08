package edu.cmu.dynet

// Implicitly convert public-facing wrapper classes to their "internal" versions.
private[dynet] object ImplicitConversions {
  implicit def dimDim(dim: Dim): internal.Dim = dim.dim
  implicit def modelModel(model: Model): internal.Model = model.model
  implicit def parameterParameter(p: Parameter): internal.Parameter = p.parameter
  implicit def lookupParameterLookupParameter(p: LookupParameter): internal.LookupParameter = p.lookupParameter
  implicit def parameterInitParameterInit(pi: ParameterInit): internal.ParameterInit = pi.parameterInit
  implicit def variableIndexVariableIndex(vi: VariableIndex): internal.SWIGTYPE_p_dynet__VariableIndex = vi.index
  implicit def floatVectorFloatVector(v: FloatVector): internal.FloatVector = v.vector
  implicit def unsignedVectorUnsignedVector(v: UnsignedVector): internal.UnsignedVector = v.vector
  implicit def expressionExpression(e: Expression): internal.Expression = e.expr
  implicit def toFloatp(fp: FloatPointer): internal.SWIGTYPE_p_float = fp.floatp
  implicit def toIntp(ip: IntPointer): internal.SWIGTYPE_p_int = ip.intp
  implicit def toUnsignedp(up: UnsignedPointer): internal.SWIGTYPE_p_unsigned_int = up.uintp
}
