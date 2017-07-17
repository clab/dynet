package edu.cmu.dynet

/** The underlying storage for a model parameter. You will never need to construct this yourself,
  * but can get it back from [[edu.cmu.dynet.ParameterCollection.parametersList()]].
  */
class ParameterStorage private[dynet](private[dynet] val storage: internal.ParameterStorage) {
  def size(): Long = storage.size()
  def dim: Dim = new Dim(storage.getDim)
  def values: Tensor = new Tensor(storage.getValues)
}

class LookupParameterStorage private[dynet](private[dynet] val storage: internal.LookupParameterStorage) {
  def size(): Long = storage.size()
}

/** A (learnable) parameter of a model.
  */
class Parameter private[dynet] (private[dynet] val parameter: internal.Parameter) {
  //def this(model: ParameterCollection, index: Long) { this(new internal.Parameter(model.model, index)) }

  def zero(): Unit = parameter.zero()

  def dim(): Dim = new Dim(parameter.dim)
  def values(): Tensor = new Tensor(parameter.values)

  def setUpdated(b: Boolean): Unit = parameter.set_updated(b)
  def isUpdated(): Boolean = parameter.is_updated()
}

class LookupParameter private[dynet] (private[dynet] val lookupParameter: internal.LookupParameter) {

  //def this(model: ParameterCollection, index: Long) { this(new internal.LookupParameter(model.model, index))}

  def initialize(index: Long, values: FloatVector) = {
    lookupParameter.initialize(index, values.vector)
  }

  def zero(): Unit = lookupParameter.zero()

  def dim(): Dim = new Dim(lookupParameter.dim)

  def setUpdated(b: Boolean): Unit = lookupParameter.set_updated(b)
  def isUpdated(): Boolean = lookupParameter.is_updated()
}

/** Specifies how to initialize a parameter. Construct a particular initialization using the
  * various factory methods on the companion object.
  */
class ParameterInit private[dynet] (
  private[dynet] val parameterInit: internal.ParameterInit) {
  def initializeParams(values: Tensor): Unit = parameterInit.initialize_params(values.tensor)
}

object ParameterInit {
  /* Initialize a parameter with random normal values */
  def normal(m: Float = 0.0f, v: Float = 1.0f): ParameterInit =
    new ParameterInit(new internal.ParameterInitNormal(m, v))

  /* Initialize a parameter with random uniform [left, right] values */
  def uniform(left: Float, right: Float): ParameterInit =
    new ParameterInit(new internal.ParameterInitUniform(left, right))

  /* Initialize a parameter with random uniform [-scale, scale] values */
  def uniform(scale: Float): ParameterInit = uniform(-scale, scale)

  /* Initialize a parameter with the constant value c */
  def const(c: Float): ParameterInit =
    new ParameterInit(new internal.ParameterInitConst(c))

  /* Initialize a parameter with the identity matrix */
  def identity(): ParameterInit =
    new ParameterInit(new internal.ParameterInitIdentity())

  /* Initialize a parameter using the specified vector of values */
  def fromVector(v: FloatVector): ParameterInit =
    new ParameterInit(new internal.ParameterInitFromVector(v.vector))

  /* Initialize a parameter using Glorot initialization */
  def glorot(isLookup: Boolean = false): ParameterInit =
    new ParameterInit(new internal.ParameterInitGlorot(isLookup))
}
