package edu.cmu.dynet


class ParameterStorage private[dynet](private[dynet] val storage: internal.ParameterStorage) {
  def size(): Long = storage.size()
  def dim: Dim = new Dim(storage.getDim)
  def values: Tensor = new Tensor(storage.getValues)
}

class LookupParameterStorage private[dynet](private[dynet] val storage: internal.LookupParameterStorage) {
  def size(): Long = storage.size()
}

class Parameter private[dynet] (private[dynet] val parameter: internal.Parameter) {
  def zero(): Unit = parameter.zero()

  def dim(): Dim = new Dim(parameter.dim)
  def values(): Tensor = new Tensor(parameter.values)

  def setUpdated(b: Boolean): Unit = parameter.set_updated(b)
  def isUpdated(): Boolean = parameter.is_updated()
}

class LookupParameter private[dynet] (private[dynet] val lookupParameter: internal.LookupParameter) {
  def initialize(index: Long, values: FloatVector) = {
    lookupParameter.initialize(index, values.vector)
  }

  def zero(): Unit = lookupParameter.zero()

  def dim(): Dim = new Dim(lookupParameter.dim)

  def setUpdated(b: Boolean): Unit = lookupParameter.set_updated(b)
  def isUpdated(): Boolean = lookupParameter.is_updated()
}

class ParameterInit private[dynet] (
  private[dynet] val parameterInit: internal.ParameterInit) {
  def initializeParams(values: Tensor): Unit = parameterInit.initialize_params(values.tensor)
}

object ParameterInit {
  def normal(m: Float = 0.0f, v: Float = 1.0f): ParameterInit =
    new ParameterInit(new internal.ParameterInitNormal(m, v))

  def uniform(left: Float, right: Float): ParameterInit =
    new ParameterInit(new internal.ParameterInitUniform(left, right))

  def uniform(scale: Float): ParameterInit = uniform(-scale, scale)

  def const(c: Float): ParameterInit =
    new ParameterInit(new internal.ParameterInitConst(c))

  def identity(): ParameterInit =
    new ParameterInit(new internal.ParameterInitIdentity())

  def fromVector(v: FloatVector): ParameterInit =
    new ParameterInit(new internal.ParameterInitFromVector(v.vector))

  def glorot(isLookup: Boolean = false): ParameterInit =
    new ParameterInit(new internal.ParameterInitGlorot(isLookup))
}
