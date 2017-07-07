package edu.cmu.dynet

/** New serialization, much less featureful than old serialization. */

class ModelSaver private[dynet](saver: internal.TextFileSaver) {
  def this(filename: String) { this(new internal.TextFileSaver(filename))}

  def addModel(model: ParameterCollection, key: String = ""): Unit = saver.save(model.model, key)
  def addParameter(p: Parameter, key: String = ""): Unit = saver.save(p.parameter, key)
  def addLookupParameter(p: LookupParameter, key: String = ""): Unit = saver.save(p.lookupParameter, key)

  def done(): Unit = saver.delete()
}

class ModelLoader private[dynet](loader: internal.TextFileLoader) {
  def this(filename: String) { this(new internal.TextFileLoader(filename))}

  def populateModel(model: ParameterCollection, key: String = ""): Unit = loader.populate(model.model, key)
  def populateParameter(p: Parameter, key: String = ""): Unit = loader.populate(p.parameter, key)
  def populateLookupParameter(p: LookupParameter, key: String = ""): Unit = loader.populate(p.lookupParameter, key)

  def done(): Unit = loader.delete()
}
