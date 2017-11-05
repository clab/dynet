package edu.cmu.dynet.examples

import edu.cmu.dynet._
import scala.language.implicitConversions

import java.nio.file.Paths

class RnnLanguageModelBatch(
  model: ParameterCollection,
  layers: Int,
  inputDim: Int,
  hiddenDim: Int,
  vocabSize: Int
) {
  val rnn = new LstmBuilder(layers, inputDim, hiddenDim, model)

  val p_c = model.addLookupParameters(vocabSize, Dim(inputDim))
  val p_R = model.addParameters(Dim(vocabSize, hiddenDim))
  val p_bias = model.addParameters(Dim(vocabSize))

  def getNegLogProb(
    sents: Seq[IntVector],
    id: Int,
    bsize: Int,
    tokens: IntPointer
  ): Expression = {
    val slen = sents(id).size
    //
    rnn.newGraph()
    //
    rnn.startNewSequence()
    //
    val i_R = Expression.parameter(p_R)
    val i_bias = Expression.parameter(p_bias)
    //
    val errs = new ExpressionVector()
    // Set all inputs to the SOS symbol
    val sos = sents(0)(0)
    var last_arr = new UnsignedVector()
    for (_ <- 1 to bsize) last_arr.add(sos)

    val next_arr = new UnsignedVector(bsize)
    // run rnn on batch
    for (t <- 1 until slen) {
      // fill next_arr
      for (i <- 0 until bsize) {
        next_arr.update(i, sents(id + i)(t))
        // count non-EOS tokens
        if (next_arr(i) != sents(id).last) tokens.increment()
      }
      // embed the current tokens
      val i_x_t = Expression.lookup(p_c, last_arr)
      //
      val i_y_t = rnn.addInput(i_x_t)
      //
      val i_r_t = i_bias + i_R * i_y_t
      //
      val i_err = Expression.pickNegLogSoftmax(i_r_t, next_arr)
      errs.add(i_err)
      // change input
      last_arr = next_arr
    }
    // add all errors
    val i_nerr = Expression.sumBatches(Expression.sum(errs))
    i_nerr
  }

  def randomSample(d: WordDict, maxLen: Int = 150, temp: Float = 1.0f) = {
    ComputationGraph.renew()
    rnn.newGraph()
    rnn.startNewSequence()
    //
    val i_R = Expression.parameter(p_R)
    val i_bias = Expression.parameter(p_bias)

    val kSOS = RnnLanguageModelBatch.kSOS
    val kEOS = RnnLanguageModelBatch.kEOS

    // start generating
    var len = 0
    var cur = kSOS
    while (len < maxLen) {
      //println("len", len, "cur", cur)
      len += 1

      val i_x_t = Expression.lookup(p_c, cur)
      //show(i_x_t.dim, "i_x_t ")
      val i_y_t = rnn.addInput(i_x_t)
      //show(i_y_t.dim, "i_y_t ")
      val i_r_t = i_bias + i_R * i_y_t
      //show(i_r_t.dim, "i_r_t ")
      val ydist = Expression.softmax(i_r_t / temp)
      //show(ydist.dim, "ydist ")

      // sample token
      var w = 0
      while (w == 0 || w == kSOS) {
        // The C++ example uses cg.incremental_forward, but that doesn't work here.
        val dist = ComputationGraph.forward(ydist)
        w = Utilities.sample(dist.toVector)
      }

      if (w == kEOS) {
        //
        rnn.startNewSequence()
        println()
        cur = kSOS
      } else {
        print(if (cur == kSOS) "" else " ")
        print(d.convert(w))
        cur = w
      }
    }
  }
}

object RnnLanguageModelBatch {
  var kSOS = 0
  var kEOS = 0

  var INPUT_VOCAB_SIZE = 0
  var OUTPUT_VOCAB_SIZE = 0

  val BATCH_SIZE = 1
  val DEV_BATCH_SIZE = 16
  val LAYERS = 2
  val INPUT_DIM = 8
  val HIDDEN_DIM = 24
  val NUM_EPOCHS = -1

  val userDir = System.getProperty("user.dir")

  val TRAIN_FILE = Paths.get(userDir, "../../examples/cpp/example-data/train-hsm.txt").toString
  val DEV_FILE = Paths.get(userDir, "../../examples/cpp/example-data/dev-hsm.txt").toString

  def main(args: Array[String]) {

    Initialize.initialize()

    val d = new WordDict
    kSOS = d.convert("<s>")
    kEOS = d.convert("</s>")

    // datasets
    val training = new scala.collection.mutable.ArrayBuffer[IntVector]
    val dev = new scala.collection.mutable.ArrayBuffer[IntVector]


    var tlc = 0
    var ttoks = 0

    for (line <- scala.io.Source.fromFile(TRAIN_FILE).getLines) {
      tlc += 1
      val row = WordDict.read_sentence(line, d)
      training.append(row)
      ttoks += row.size
    }
    println(s"${tlc} lines, ${ttoks} tokens, ${d.size} types")

    // sort the training sentences in descending order of length (for minibatching)
    training.sortBy(row => -row.size).zipWithIndex.foreach {
      case (iv, i) => training(i) = iv
    }

    // pad the sentences in the same batch with EOS so they are the same length
    for (i <- 0 until training.size by BATCH_SIZE) {
      for (j <- 1 until BATCH_SIZE) {
        while (training(i + j).size < training(i).size) {
          training(i + j).add(kEOS)
        }
      }
    }

    // freeze dictionary
    d.freeze()
    d.set_unk("UNK")

    INPUT_VOCAB_SIZE = d.size()
    OUTPUT_VOCAB_SIZE = d.size()

    // read validation dataset
    var dlc = 0
    var dtoks = 0

    for (line <- scala.io.Source.fromFile(DEV_FILE).getLines) {
      dlc += 1
      val row = WordDict.read_sentence(line, d)
      dev.append(row)
      dtoks += row.size
    }
    println(s"${dlc} lines, ${dtoks} tokens")

    // sort the dev sentences in descending order of length
    dev.sortBy(row => -row.size).zipWithIndex.foreach {
      case (iv, i) => dev(i) = iv
    }

    // pad
    for (i <- 0 until dev.size by BATCH_SIZE) {
      for (j <- 1 until BATCH_SIZE) {
        while (dev(i + j).size < dev(i).size) {
          dev(i + j).add(kEOS)
        }
      }
    }

    val model = new ParameterCollection
    val adam = new AdamTrainer(model, 0.001f, 0.9f, 0.999f, 1e-8f)
    adam.clipThreshold = adam.clipThreshold * BATCH_SIZE

    val lm = new RnnLanguageModelBatch(model, LAYERS, INPUT_DIM, HIDDEN_DIM, INPUT_VOCAB_SIZE)

    val numBatches = training.size / BATCH_SIZE - 1
    val numDevBatches = dev.size / DEV_BATCH_SIZE - 1

    val sizeSamples = 200

    // random indexing
    val order = new IntVector(0 until numBatches)

    var first = true
    var epoch = 0

    while (epoch < NUM_EPOCHS || NUM_EPOCHS < 0) {
      //
      if (first) {
        first = false
      } else {
        adam.updateEpoch()
      }
      // reshuffle
      Utilities.shuffle(order)

      var loss = 0.0
      val tokens = new IntPointer

      for (si <- 0 until numBatches) {
        ComputationGraph.renew
        val id = order(si) * BATCH_SIZE
        val bsize = math.min(training.size - id, BATCH_SIZE)
        val loss_expr = lm.getNegLogProb(training, id, bsize, tokens)

        loss += ComputationGraph.forward(loss_expr).toFloat

        ComputationGraph.backward(loss_expr)

        adam.update()
        //
        if ((si + 1) % (numBatches / 10) == 0 || si == numBatches - 1) {
          adam.status()
          val lt = loss / tokens.value
          println(s" E = ${lt} ppl=${math.exp(lt)}")

          loss = 0.0
          tokens.set(0)
        }
      }

      var dloss = 0.0
      val dtokens = new IntPointer
      for (i <- 0 until numDevBatches) {
        ComputationGraph.renew

        val id = i * DEV_BATCH_SIZE
        val bsize = math.min(dev.size - id, DEV_BATCH_SIZE)

        val loss_expr = lm.getNegLogProb(dev, id, bsize, dtokens)

        dloss += ComputationGraph.forward(loss_expr).toFloat
      }

      val dt = dloss / dtokens.value
      println(s"***DEV [epoch=${epoch}] E = ${dt} ppl=${math.exp(dt)}")

      println("-----------------------")
      lm.randomSample(d, sizeSamples)
      println("-----------------------")
      epoch += 1
    }
  }
}
