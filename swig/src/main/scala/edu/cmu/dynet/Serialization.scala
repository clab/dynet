package edu.cmu.dynet


class ModelSaver private[dynet](saver: internal.ModelSaver) {
  def this(filename: String) { this(new internal.ModelSaver(filename)) }

  def addModel(model: Model): Unit = saver.add_model(model.model)
  def addParameter(p: Parameter): Unit = saver.add_parameter(p.parameter)
  def addLookupParameter(p: LookupParameter): Unit = saver.add_lookup_parameter(p.lookupParameter)
  def addLSTMBuilder(p: LSTMBuilder): Unit = saver.add_lstm_builder(p.builder)
  //def addVanillaLSTMBuilder
  def addSRNNBuilder(p: SimpleRNNBuilder): Unit = saver.add_srnn_builder(p.builder)
  // def addGRUBuilder
  // def addFastLSTMBuilder

  def addSize(len: Long): Unit = saver.add_size(len)
  def addByteArray(bytes: Array[Byte]): Unit = saver.add_byte_array(bytes)

  def addInt(x: Int): Unit = saver.add_int(x)
  def addLong(x: Long): Unit = saver.add_long(x)
  def addFloat(x: Float): Unit = saver.add_float(x)
  def addDouble(x: Double): Unit = saver.add_double(x)
  def addBoolean(x: Boolean): Unit = saver.add_boolean(x)

  def addObject(x: java.io.Serializable) = saver.add_object(x)

  def done(): Unit = saver.done()
}

class ModelLoader private[dynet](loader: internal.ModelLoader) {
  def this(filename: String) { this(new internal.ModelLoader(filename)) }

  def loadModel(): Model = new Model(loader.load_model())
  def loadParameter(): Parameter = new Parameter(loader.load_parameter())
  def loadLookupParameter(): LookupParameter = new LookupParameter(loader.load_lookup_parameter())
  def loadLSTMBuilder(): LSTMBuilder = new LSTMBuilder(loader.load_lstm_builder())
  //def loadVanillaLSTMBuilder
  def loadSRNNBuilder(): SimpleRNNBuilder = new SimpleRNNBuilder(loader.load_srnn_builder())
  // def loadGRUBuilder
  // def loadFastLSTMBuilder

  def loadSize(): Long = loader.load_size()
  def loadByteArray(buffer: Array[Byte]): Unit = loader.load_byte_array(buffer)

  def loadInt(): Int = loader.load_int()
  def loadLong(): Long = loader.load_long()
  def loadFloat(): Float = loader.load_float()
  def loadDouble(): Double = loader.load_double()
  def loadBoolean(): Boolean = loader.load_boolean()

  def loadObject[T](clazz: Class[T]): T = loader.load_object(clazz)

  def done(): Unit = loader.done()
}

