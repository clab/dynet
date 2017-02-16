package edu.cmu.dynet

import org.scalatest._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._
import java.util.Arrays

class SerializationSpec extends FlatSpec with Matchers {
  import DynetScalaHelpers._

  myInitialize()

  def assertSameModel(m1: Model, m2: Model): Unit = {
    // TODO(joelgrus): add more logic here as we add more methods to the Java API
    m1.parameter_count() shouldBe m2.parameter_count()

    val parameters1 = m1.parameters_list()
    val parameters2 = m2.parameters_list()

    parameters1.size() shouldBe parameters2.size()

    parameters1.zip(parameters2).foreach {
      case (p1, p2) => {
        p1.size shouldBe p2.size
        p1.getDim shouldBe p2.getDim
        p1.getValues.toSeq shouldBe p2.getValues.toSeq
      }
    }

    val lookupParameters1 = m1.lookup_parameters_list()
    val lookupParameters2 = m2.lookup_parameters_list()

    lookupParameters1.size() shouldBe lookupParameters2.size()

    lookupParameters1.zip(lookupParameters2).foreach {
      case (p1, p2) => {
        p1.size shouldBe p2.size
      }
    }
  }

  def defaultModel(): Model = {
    val model = new Model()
    model.add_parameters(dim(2, 3))
    model.add_parameters(dim(5))
    model
  }

  "dynet" should "create models with the right number of parameters" in {
    val model = defaultModel()
    model.parameter_count() shouldBe 11  // == 2 * 3 + 5
  }

  "dynet" should "serialize models to/from disk" in {
    val original = defaultModel()

    // Save to a temp file
    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    save_dynet_model(path, original)

    // Deserialize
    val deserialized = new Model()
    load_dynet_model(path, deserialized)

    assertSameModel(original, deserialized)
  }

  "dynet" should "serialize models to/from string when called explicitly" in {

    val original = defaultModel()

    val s = original.serialize_to_string()

    val deserialized = new Model
    deserialized.load_from_string(s)
    assertSameModel(original, deserialized)
  }

  "dynet" should "correctly implement java serialization" in {

    val original = defaultModel()

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(path))
    oos.writeObject(original)
    oos.close()

    val ois = new java.io.ObjectInputStream(new java.io.FileInputStream(path))
    val deserialized = ois.readObject.asInstanceOf[Model]
    assertSameModel(original, deserialized)
  }

  "model saver and model loader" should "handle simplernnbuilder correctly" in {

    // this is the simple_rnn_io test case from the C++ tests
    val mod1 = new Model()
    val rnn1 = new SimpleRNNBuilder(1, 10, 10, mod1)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.add_model(mod1)
    saver.add_srnn_builder(rnn1)
    saver.done()

    val loader = new ModelLoader(path)
    val mod2 = loader.load_model()
    val rnn2 = loader.load_srnn_builder()
    loader.done()

    assertSameModel(mod1, mod2)
  }

  "model saver and model loader" should "handle vanillalstmbuilder correctly" in {

    // this is the vanilla_lstm_io test case from the C++ tests
    val mod1 = new Model()
    val rnn1 = new VanillaLSTMBuilder(1, 10, 10, mod1)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.add_model(mod1)
    saver.add_vanilla_lstm_builder(rnn1)
    saver.done()

    val loader = new ModelLoader(path)
    val mod2 = loader.load_model()
    val rnn2 = loader.load_vanilla_lstm_builder()
    loader.done()

    assertSameModel(mod1, mod2)
  }


  "model saver and model loader" should "handle lstmbuilder correctly" in {

    // this is the lstm_io test case from the C++ tests
    val mod1 = new Model()
    val rnn1 = new LSTMBuilder(1, 10, 10, mod1)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.add_model(mod1)
    saver.add_lstm_builder(rnn1)
    saver.done()

    val loader = new ModelLoader(path)
    val mod2 = loader.load_model()
    val rnn2 = loader.load_lstm_builder()
    loader.done()

    assertSameModel(mod1, mod2)
  }
  
  "model saver and model loader" should "handle byte[] correctly" in {

    val s = Array[Byte](3, 7, 127, 2, 5, 8, 0, -1, -2, 100, 10, -2, 0)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.add_size(s.length)
    saver.add_byte_array(s)
    saver.done()

    val loader = new ModelLoader(path)
    val length = loader.load_size()
    val s2 = Array.ofDim[Byte](length.asInstanceOf[Int])
    loader.load_byte_array(s2)
    loader.done()

    Arrays.equals(s, s2) shouldBe true
  }

  "model saver and model loader" should "handle objects correctly" in {
    val s = new Foo("abcd", 78)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.add_object(s)
    saver.done()

    val loader = new ModelLoader(path)
    val s2 = loader.load_object(classOf[Foo])
    loader.done()

    s2 shouldBe s
  }

  "model saver and model loader" should "handle primitives correctly" in {
    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.add_int(0)
    saver.add_int(-123)
    saver.add_int(256)
    saver.add_long(-23L)
    saver.add_float(256.123f)
    saver.add_double(-12.54)
    saver.add_boolean(true)
    saver.add_boolean(false)
    saver.done()

    val loader = new ModelLoader(path)
    loader.load_int() shouldBe 0
    loader.load_int() shouldBe -123
    loader.load_int() shouldBe 256
    loader.load_long() shouldBe -23L
    loader.load_float() shouldBe 256.123f
    loader.load_double() shouldBe -12.54
    loader.load_boolean() shouldBe true
    loader.load_boolean() shouldBe false
    loader.done()
  }
}

case class Foo(a: String, b: Int) extends Serializable
