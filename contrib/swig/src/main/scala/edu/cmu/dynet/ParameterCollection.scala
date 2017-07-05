package edu.cmu.dynet

class ParameterCollection private[dynet] (private[dynet] val model: internal.ParameterCollection) {

  def this() { this( new internal.ParameterCollection ) }

  def gradientL2Norm(): Float = model.gradient_l2_norm()
  def resetGradient(): Unit = model.reset_gradient()

  /*
  def getParameter(index: Long): Parameter =
    new Parameter(new internal.Parameter(this.model, index))
  def getLookupParameter(index: Long): LookupParameter = {
    new LookupParameter(new internal.LookupParameter(this.model, index))
  }
  */

  def addParameters(d: Dim, scale: Float = 0.0f): Parameter =
    new Parameter(model.add_parameters(d.dim, scale))
  def addParameters(d: Dim, init: ParameterInit): Parameter =
    new Parameter(model.add_parameters(d.dim, init.parameterInit))

  def addLookupParameters(n: Long, d: Dim): LookupParameter =
    new LookupParameter(model.add_lookup_parameters(n, d.dim))
  def addLookupParameters(n: Long, d: Dim, init: ParameterInit) =
    new LookupParameter(model.add_lookup_parameters(n, d.dim, init.parameterInit))

  def projectWeights(radius: Float = 0.0f) = model.project_weights(radius)
  def setWeightDecayLambda(lambda: Float) = model.set_weight_decay_lambda(lambda)

  def parameterCount(): Long = model.parameter_count()
  def updatedParameterCount(): Long = model.updated_parameter_count()

  /*
  def setUpdatedParam(p: Parameter, status: Boolean): Unit =
    model.set_updated_param(p.parameter, status)
  def setUpdatedLookupParam(p: LookupParameter, status: Boolean): Unit =
    model.set_updated_lookup_param(p.lookupParameter, status)

  def isUpdatedParam(p: Parameter) = model.is_updated_param(p.parameter)
  def isUpdatedLookupParam(p: LookupParameter) = model.is_updated_lookup_param(p.lookupParameter)
  */

  def parametersList(): Seq[ParameterStorage] = {
    val params = model.parameters_list
    for (i <- 0 until params.size.toInt) yield new ParameterStorage(params.get(i))
  }

  def lookupParametersList(): Seq[LookupParameterStorage] = {
    val params = model.lookup_parameters_list
    for (i <- 0 until params.size.toInt) yield new LookupParameterStorage(params.get(i))
  }
}
