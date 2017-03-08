package edu.cmu.dynet

import ImplicitConversions._

object ComputationGraph {
  private[dynet] var cg: internal.ComputationGraph = internal.ComputationGraph.getNew
  var version: Long = 0L

  def renew(): Unit = {
    cg = internal.ComputationGraph.getNew
    version += 1
  }

  def addInput(s: Float): VariableIndex = new VariableIndex(cg.add_input(s))
  def addInput(d: Dim, data: FloatVector): VariableIndex = new VariableIndex(cg.add_input(d, data))
  def addInput(d: Dim, ids: UnsignedVector, data: FloatVector, defdata: Float = 0.0f) =
    new VariableIndex(cg.add_input(d, ids, data, defdata))

  def addParameters(p: Parameter): VariableIndex = new VariableIndex(cg.add_parameters(p))
  def addConstParameters(p: Parameter): VariableIndex = new VariableIndex(cg.add_const_parameters(p))

  def addLookup(p: LookupParameter, pindex: UnsignedPointer): VariableIndex =
    new VariableIndex(cg.add_lookup(p, pindex))
  def addLookup(p: LookupParameter, index: Long): VariableIndex =
    new VariableIndex(cg.add_lookup(p, index))
  def addLookup(p: LookupParameter, indices: UnsignedVector): VariableIndex =
    new VariableIndex(cg.add_lookup(p, indices))

  def addConstLookup(p: LookupParameter, pindex: UnsignedPointer): VariableIndex =
    new VariableIndex(cg.add_const_lookup(p, pindex))
  def addConstLookup(p: LookupParameter, index: Long): VariableIndex =
    new VariableIndex(cg.add_const_lookup(p, index))
  def addConstLookup(p: LookupParameter, indices: UnsignedVector): VariableIndex =
    new VariableIndex(cg.add_const_lookup(p, indices))

  def getDimension(index: VariableIndex): Dim = new Dim(cg.get_dimension(index))

  def clear(): Unit = cg.clear()
  def checkpoint(): Unit = cg.checkpoint()
  def revert(): Unit = cg.revert()

  def forward(last: Expression): Tensor = new Tensor(cg.forward(last))
  def incrementalForward(last: Expression): Tensor = new Tensor(cg.incremental_forward(last))
  def getValue(e: Expression): Tensor = new Tensor(cg.get_value(e))

  def invalidate(): Unit = cg.invalidate()
  def backward(last: Expression): Unit = cg.backward(last)

  def printGraphViz(): Unit = cg.print_graphviz()
}