package edu.cmu.dynet

/** The ComputationGraph object contains the singleton DyNet computation graph instance. Any C++
  * instance method is instead implemented as a static function here.*
  */
object ComputationGraph {
  private[dynet] var cg: internal.ComputationGraph = internal.ComputationGraph.getNew
  var version: Long = 0L
  private var defaultDevice: internal.Device = internal.dynet_swig.getDefault_device()

  /** Gets rid of the singleton Computation Graph and replaces it with a fresh one. Increments
    * `version` to make sure we don't use any stale expressions.
    */
  def renew(): Unit = {
    cg = internal.ComputationGraph.getNew
    version += 1
  }

  def addInput(s: Float): VariableIndex = new VariableIndex(cg.add_input(s, defaultDevice))
  def addInput(d: Dim, data: FloatVector): VariableIndex =
    new VariableIndex(cg.add_input(d.dim, data.vector, defaultDevice))
  def addInput(d: Dim, ids: UnsignedVector, data: FloatVector, defdata: Float = 0.0f) =
    new VariableIndex(cg.add_input(d.dim, ids.vector, data.vector, defaultDevice, defdata))

  def addParameters(p: Parameter): VariableIndex = new VariableIndex(cg.add_parameters(p.parameter))
  def addConstParameters(p: Parameter): VariableIndex =
    new VariableIndex(cg.add_const_parameters(p.parameter))

  def addLookup(p: LookupParameter, pindex: UnsignedPointer): VariableIndex =
    new VariableIndex(cg.add_lookup(p.lookupParameter, pindex.uintp))
  def addLookup(p: LookupParameter, index: Long): VariableIndex =
    new VariableIndex(cg.add_lookup(p.lookupParameter, index))
  def addLookup(p: LookupParameter, indices: UnsignedVector): VariableIndex =
    new VariableIndex(cg.add_lookup(p.lookupParameter, indices.vector))

  def addConstLookup(p: LookupParameter, pindex: UnsignedPointer): VariableIndex =
    new VariableIndex(cg.add_const_lookup(p.lookupParameter, pindex.uintp))
  def addConstLookup(p: LookupParameter, index: Long): VariableIndex =
    new VariableIndex(cg.add_const_lookup(p.lookupParameter, index))
  def addConstLookup(p: LookupParameter, indices: UnsignedVector): VariableIndex =
    new VariableIndex(cg.add_const_lookup(p.lookupParameter, indices.vector))

  def getDimension(index: VariableIndex): Dim = new Dim(cg.get_dimension(index.index))

  def clear(): Unit = cg.clear()
  def checkpoint(): Unit = cg.checkpoint()
  def revert(): Unit = cg.revert()

  def forward(last: Expression): Tensor = new Tensor(cg.forward(last.expr))
  def incrementalForward(last: Expression): Tensor = new Tensor(cg.incremental_forward(last.expr))
  def getValue(e: Expression): Tensor = new Tensor(cg.get_value(e.expr))

  def invalidate(): Unit = cg.invalidate()
  def backward(last: Expression): Unit = cg.backward(last.expr)

  def printGraphViz(): Unit = cg.print_graphviz()
}
