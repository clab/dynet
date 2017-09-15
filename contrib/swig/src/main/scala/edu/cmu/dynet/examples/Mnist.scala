package edu.cmu.dynet.examples

import edu.cmu.dynet._

import scala.language.implicitConversions

object MnistFile {
  // I basically reverse engineered the file formats from java/org/deeplearning4j/datasets/mnist

  def getImages(filename: String): Seq[FloatVector] = {
    val fis = new java.io.FileInputStream(filename)
    val stream = new java.io.DataInputStream(fis)

    // check magic number
    assert(2051 == stream.readInt)

    val count = stream.readInt()
    val rows = stream.readInt()
    val cols = stream.readInt()

    def nextImage(): FloatVector = new FloatVector(
      for (row <- 1 to rows; col <- 1 to cols) yield stream.readUnsignedByte().toFloat
    )

    for (i <- 1 to count) yield nextImage()
  }

  def getLabels(filename: String): Seq[Int] = {
    val fis = new java.io.FileInputStream(filename)
    val stream = new java.io.DataInputStream(fis)

    // check magic number
    assert(2049 == stream.readInt)

    val count = stream.readInt

    for (i <- 1 to count) yield stream.readUnsignedByte
  }
}

object Mnist {

  val usage = """sbt \"runMain edu.cmu.dynet.examples.Mnist --train train-images.idx3-ubyte --train_labels train-labels.idx1-ubyte --dev t10k-images.idx3-ubyte --dev_labels t10k-labels.idx1-ubyte --batch_size 128 --num_epochs 20\""""

  def parseArgs(
    args: List[String],
    oldMap: Map[String, String] = Map.empty[String, String]
  ): Map[String, String] = args match {
    case Nil => oldMap
    case k :: v :: rest => parseArgs(rest, oldMap.updated(k, v))
    case _ => throw new IllegalArgumentException("usage")
  }

  def main(args: Array[String]) {
    Initialize.initialize()

    val argList = args.toList
    val params = parseArgs(argList)
    val batchSize = params.get("--batch_size").map(_.toInt).getOrElse(128)
    val numEpochs = params.get("--num_epochs").map(_.toInt).getOrElse(20)

    val model = new ParameterCollection()
    val adam = new AdamTrainer(model)
    adam.clipThreshold = adam.clipThreshold * batchSize

    // create model
    val nn = new MultiLayerPerceptron(model, Seq(
      new Layer(784, 512, Activation.RELU,   0.2f),
      new Layer(512, 512, Activation.RELU,   0.2f),
      new Layer(512, 10,  Activation.LINEAR, 0.0f)
    ))

    // Initialize variables for training

    val train = MnistFile.getImages(params("--train"))
    val trainLabels = MnistFile.getLabels(params("--train_labels"))
    val trainCount = train.size
    println(s"${trainCount} training images")

    val dev = MnistFile.getImages(params("--dev"))
    val devLabels = MnistFile.getLabels(params("--dev_labels"))
    val devCount = dev.size
    println(s"${devCount} dev images")

    val numBatches = trainCount / batchSize - 1

    // random indexing
    val order = new IntVector(0 until numBatches)

    var first = true
    var epoch = 0

    while (epoch < numEpochs || numEpochs < 0) {
      // update the optimizer
      if (first) { first = false } else { adam.updateEpoch() }
      //reshuffle
      Utilities.shuffle(order)

      var loss = 0.0
      var numSamples = 0.0

      nn.enable_dropout()

      for (si <- 0 until numBatches) {
        // build graph
        ComputationGraph.renew()

        // get the current batch of images and labels
        val id = order(si) * batchSize
        val bsize = math.min(trainCount - id, batchSize)

        val curBatch = new ExpressionVector(
          for (image <- train.slice(id, id + bsize))
            yield Expression.input(Dim(784), image)
        )
        val curLabels = new UnsignedVector(trainLabels.slice(id, id + bsize).map(_.toLong))

        // reshape as batch
        val xBatch = Expression.reshape(Expression.concatenateCols(curBatch), Dim(Seq(784), bsize))
        // get negative log likelihood on batch
        val lossExpr = nn.get_nll(xBatch, curLabels)
        // get scalar error for monitoring
        loss += ComputationGraph.forward(lossExpr).toFloat
        // increment number of samples processed
        numSamples += bsize
        // compute gradient
        ComputationGraph.backward(lossExpr)
        // update parameters
        adam.update()
        // print progress every 10th of the dataset
        if ((si + 1) % (numBatches / 10) == 0 || si == numBatches - 1) {
          adam.status()
          println(s" E = ${loss / numSamples}")
          loss = 0.0
          numSamples = 0.0
        }
      }

      // disable dropouts for dev testing
      nn.disable_dropout()

      // show score on dev data
      if (true) {
        var dpos = 0.0
        for ((image, label) <- dev.zip(devLabels)) {
          ComputationGraph.renew()
          val x = Expression.input(Dim(784), image)
          val predictedIdx = nn.predict(x)

          // increment count of positive classification
          if (predictedIdx == label) {
            dpos += 1
          }
        }
        println(s"***DEV [epoch=${epoch}] E = ${dpos / devCount}")
      }
      epoch += 1
    }
  }
}
