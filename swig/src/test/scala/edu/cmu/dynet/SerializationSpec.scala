package edu.cmu.dynet

import org.scalatest._
import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

class SerializationSpec extends FlatSpec with Matchers {
  import DynetScalaHelpers._

  myInitialize()

  "dynet" should "correctly serialize and deserialize" in {
    val original = new Model()

    // Add parameters
    original.add_parameters(dim(2, 3))
    original.add_parameters(dim(5))

    // Save to a temp file
    val path = java.io.File.createTempFile("dynet_test", "serialization_spec").getAbsolutePath
    save_dynet_model(path, original)

    // Deserialize
    val deserialized = new Model()
    load_dynet_model(path, deserialized)
    deserialized.parameter_count() shouldBe 11  // == 2 * 3 + 5
  }
}
