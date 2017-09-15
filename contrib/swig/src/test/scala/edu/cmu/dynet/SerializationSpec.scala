package edu.cmu.dynet

import org.scalatest._
import Matchers._
import edu.cmu.dynet.internal.{dynet_swig => dn}
import java.util.Arrays

class SerializationSpec extends FlatSpec with Matchers {
  Initialize.initialize()

  def assertSameSeq(s1: Seq[Float], s2: Seq[Float], eps: Float = 1e-5f): Unit = {
    s1.size shouldBe s2.size
    s1.zip(s2).foreach { case (v1, v2) => v1 shouldBe v2 +- eps }
  }

  def assertSameModel(m1: ParameterCollection, m2: ParameterCollection): Unit = {
    // TODO(joelgrus): add more logic here as we add more methods to the Java API
    m1.parameterCount() shouldBe m2.parameterCount()

    val parameters1 = m1.parametersList()
    val parameters2 = m2.parametersList()

    parameters1.size shouldBe parameters2.size

    parameters1.zip(parameters2).foreach {
      case (p1, p2) => {
        p1.size shouldBe p2.size
        p1.dim shouldBe p2.dim
        assertSameSeq(p1.values.toSeq, p2.values.toSeq)
      }
    }

    val lookupParameters1 = m1.lookupParametersList()
    val lookupParameters2 = m2.lookupParametersList()

    lookupParameters1.size shouldBe lookupParameters2.size

    lookupParameters1.zip(lookupParameters2).foreach {
      case (p1, p2) => {
        p1.size shouldBe p2.size
      }
    }
  }

  def defaultModel(): ParameterCollection = {
    val model = new ParameterCollection()
    model.addParameters(Dim(2, 3))
    model.addParameters(Dim(5))
    model
  }


  "dynet" should "create models with the right number of parameters" in {
    val model = defaultModel()
    model.parameterCount() shouldBe 11  // == 2 * 3 + 5
  }

  "model saver and model loader" should "handle simplernnbuilder correctly" in {

    // this is the simple_rnn_io test case from the C++ tests
    val mod1 = new ParameterCollection()
    val rnn1 = new SimpleRnnBuilder(1, 10, 10, mod1)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath

    val saver = new ModelSaver(path)
    saver.addModel(mod1)
    saver.done()

    val loader = new ModelLoader(path)
    val mod2 = new ParameterCollection()
    val rnn2 = new SimpleRnnBuilder(1, 10, 10, mod2)

    loader.populateModel(mod2)
    assertSameModel(mod1, mod2)
  }


  "model saver and model loader" should "handle vanillalstmbuilder correctly" in {

    // this is the vanilla_lstm_io test case from the C++ tests
    val mod1 = new ParameterCollection()
    val rnn1 = new VanillaLstmBuilder(1, 10, 10, mod1)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.addModel(mod1)
    saver.done()

    val loader = new ModelLoader(path)
    val mod2 = new ParameterCollection()
    val rnn2 = new VanillaLstmBuilder(1, 10, 10, mod2)

    loader.populateModel(mod2)
    assertSameModel(mod1, mod2)
  }

  // TODO(joelgrus): implement these
  /*


  "model saver and model loader" should "handle byte[] correctly" in {

    val s = Array[Byte](3, 7, 127, 2, 5, 8, 0, -1, -2, 100, 10, -2, 0)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.addSize(s.length)
    saver.addByteArray(s)
    saver.done()

    val loader = new ModelLoader(path)
    val length = loader.loadSize()
    val s2 = Array.ofDim[Byte](length.asInstanceOf[Int])
    loader.loadByteArray(s2)
    loader.done()

    Arrays.equals(s, s2) shouldBe true
  }

  "model saver and model loader" should "handle objects correctly" in {
    val s = new Foo("abcd", 78)

    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.addObject(s)
    saver.done()

    val loader = new ModelLoader(path)
    val s2 = loader.loadObject(classOf[Foo])
    loader.done()

    s2 shouldBe s
  }

  "model saver and model loader" should "handle primitives correctly" in {
    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    val saver = new ModelSaver(path)
    saver.addInt(0)
    saver.addInt(-123)
    saver.addInt(256)
    saver.addLong(-23L)
    saver.addFloat(256.123f)
    saver.addDouble(-12.54)
    saver.addBoolean(true)
    saver.addBoolean(false)
    saver.done()

    val loader = new ModelLoader(path)
    loader.loadInt() shouldBe 0
    loader.loadInt() shouldBe -123
    loader.loadInt() shouldBe 256
    loader.loadLong() shouldBe -23L
    loader.loadFloat() shouldBe 256.123f
    loader.loadDouble() shouldBe -12.54
    loader.loadBoolean() shouldBe true
    loader.loadBoolean() shouldBe false
    loader.done()
  }
*/

}

case class Foo(a: String, b: Int) extends Serializable
