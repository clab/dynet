package edu.cmu.dynet

import ImplicitConversions._

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
    new ParameterInit(new internal.ParameterInitFromVector(v))

  def glorot(isLookup: Boolean = false): ParameterInit =
    new ParameterInit(new internal.ParameterInitGlorot(isLookup))
}


class Model private[dynet] (private[dynet] val model: internal.Model) {

  def this() { this( new internal.Model ) }

  def gradientL2Norm(): Float = model.gradient_l2_norm()
  def resetGradient(): Unit = model.reset_gradient()

  def getParameter(index: Long): Parameter =
    new Parameter(new internal.Parameter(this.model, index))
  def getLookupParameter(index: Long): LookupParameter = {
    new LookupParameter(new internal.LookupParameter(this.model, index))
  }

  def addParameters(d: Dim, scale: Float = 0.0f): Parameter =
    new Parameter(model.add_parameters(d, scale))
  def addParameters(d: Dim, init: ParameterInit): Parameter =
    new Parameter(model.add_parameters(d, init))

  def addLookupParameters(n: Long, d: Dim): LookupParameter =
    new LookupParameter(model.add_lookup_parameters(n, d))
  def addLookupParameters(n: Long, d: Dim, init: ParameterInit) =
    new LookupParameter(model.add_lookup_parameters(n, d, init))

  def projectWeights(radius: Float = 0.0f) = model.project_weights(radius)
  def setWeightDecayLambda(lambda: Float) = model.set_weight_decay_lambda(lambda)

  def setUpdatedParam(p: Parameter, status: Boolean): Unit = model.set_updated_param(p, status)
  def setUpdatedLookupParam(p: LookupParameter, status: Boolean): Unit =
    model.set_updated_lookup_param(p, status)

  def isUpdatedParam(p: Parameter) = model.is_updated_param(p)
  def isUpdatedLookupParam(p: LookupParameter) = model.is_updated_lookup_param(p)
}
