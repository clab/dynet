package edu.cmu.dynet

import org.scalatest._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class SerializationSpec extends FlatSpec with Matchers {
  import DynetScalaHelpers._

  myInitialize()

  def assertSameModel(m1: Model, m2: Model): Unit = {
    // TODO(joelgrus): add more logic here as we add more methods to the Java API
    m1.parameter_count() shouldBe m2.parameter_count()

    val parameters1 = m1.parameters_list()
    val parameters2 = m2.parameters_list()

    parameters1.zip(parameters2).foreach {
      case (p1, p2) => {
        p1.size shouldBe p2.size
        p1.getDim shouldBe p2.getDim
        p1.getValues.toSeq shouldBe p2.getValues.toSeq
      }
    }

    val lookupParameters1 = m1.lookup_parameters_list()
    val lookupParameters2 = m2.lookup_parameters_list()

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
}
