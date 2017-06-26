package edu.cmu.dynet.examples

import edu.cmu.dynet._

import scala.language.implicitConversions

import java.nio.file.Paths

class EncoderDecoder(
  model: ParameterCollection,
  numLayers: Int,
  inputDim: Int,
  hiddenDim: Int,
  bidirectional: Boolean = false) {

  val decBuilder = new LstmBuilder(numLayers, inputDim, hiddenDim, model)
  val fwdEncBuilder = new LstmBuilder(numLayers, inputDim, hiddenDim, model)

  var revEncBuilder: Option[RnnBuilder] = None
  var p_ie2oe: Option[Parameter] = None
  var p_boe: Option[Parameter] = None

  if (bidirectional) {
    revEncBuilder = Some(new LstmBuilder(numLayers, inputDim, hiddenDim, model))
    p_ie2oe = Some(model.addParameters(Dim(hiddenDim * numLayers * 2, hiddenDim * numLayers * 4)))
    p_boe = Some(model.addParameters(Dim(hiddenDim * numLayers * 2)))
  }

  val p_c = model.addLookupParameters(EncoderDecoder.INPUT_VOCAB_SIZE, Dim(inputDim))
  val p_ec = model.addLookupParameters(EncoderDecoder.INPUT_VOCAB_SIZE, Dim(inputDim))
  val p_R = model.addParameters(Dim(EncoderDecoder.OUTPUT_VOCAB_SIZE, hiddenDim))
  val p_bias = model.addParameters(Dim(EncoderDecoder.OUTPUT_VOCAB_SIZE))


  def encode(isents: Seq[IntVector],
             id: Int,
             bsize: Int,
             chars: IntPointer): Expression = {
    val islen = isents(id).size
    val x_t = new UnsignedVector(bsize)

    // Forward encoder --------

    // Initialize parameters in fwd_enc_builder
    fwdEncBuilder.newGraph()
    // Initialize the sequence
    fwdEncBuilder.startNewSequence()

    // Run the forward encoder on the batch
    for (t <- 0 until islen) {
      // Fill x_t with the characters at step t in the batch
      for (i <- 0 until bsize) {
        x_t.update(i, isents(id+i)(t))
        if (x_t(i) != isents(id)(islen - 1)) {
          // if x_t is non-EOS, count a char
          chars.set(chars.value() + 1)
        }
      }

      // Get embedding
      val i_x_t = Expression.lookup(p_ec, x_t)
      // Run a step in the forward encoder
      fwdEncBuilder.addInput(i_x_t)
    }

    // Backward encoder --------------
    if (bidirectional) {
      // Initialize parameters in bwd_enc_builder
      revEncBuilder.get.newGraph()
      // Initialize the sequence
      revEncBuilder.get.startNewSequence()
      // Fill x_t with the characters at step t in the batch
      for (t <- (0 until islen).reverse) {
        for (i <- 0 until bsize) {
          x_t.update(i, isents(id+i)(t))
        }
        // Get embedding
        val i_x_t = Expression.lookup(p_ec, x_t)
        // Run a step in the reverse encoder
        revEncBuilder.get.addInput(i_x_t)
      }
    }

    // Collect encodings -------
    val to = new ExpressionVector()
    // Get states from forward encoder
    fwdEncBuilder.finalS().foreach(to.add)
    // Get states from backward encoder
    if (bidirectional) {
      revEncBuilder.get.finalS.foreach(to.add)
    }

    // Put it as a vector
    val i_combined = Expression.concatenate(to)
    val i_nc = if (bidirectional) {
      // Perform an affine transformation for rescaling
      val i_ie2oe = Expression.parameter(p_ie2oe.get)
      val i_bie = Expression.parameter(p_boe.get)
      i_bie + i_ie2oe * i_combined
    } else {
      i_combined
    }

    i_nc
  }

  // Single sentence version
  def encode(insent: IntVector): Expression = {
    val isents = Seq(insent)
    val chars = new IntPointer
    chars.set(0)
    encode(isents, 0, 1, chars)
  }

  // Batched decoding
  def decode(i_nc: Expression,
    osents: Seq[IntVector],
    id: Int,
    bsize: Int): Expression = {
    // Reconstruct input states from encodings ---------
    // List of input states for decoder
    val oein = new ExpressionVector
    // Add input cell states
    for (i <- 0 until numLayers) {
      oein.add(Expression.pickrange(i_nc, i * hiddenDim, (i+1) * hiddenDim))
    }
    // Add input output states
    for (i <- 0 until numLayers) {
      oein.add(Expression.pickrange(i_nc,
        hiddenDim * numLayers + i * hiddenDim,
        hiddenDim * numLayers + (i+1) * hiddenDim))
    }

    // Initialize graph for decoder
    decBuilder.newGraph()
    // Initialize new sequence with encoded states
    decBuilder.startNewSequence(oein)

    // Run decoder ------------------
    // Add parameters to the graph
    val i_R = Expression.parameter(p_R)
    val i_bias = Expression.parameter(p_bias)
    // Initialize errors and input vectors
    val errs = new ExpressionVector
    var x_t = new UnsignedVector(bsize)

    // Set start of sequence
    for (i <- 0 until bsize) {
      x_t.update(i, osents(id + i)(0))
    }
    val next_x_t = new UnsignedVector(bsize)

    val oslen = osents(id).size

    // Run on output sentence
    for (t <- 1 until oslen) {
      // Retrieve input
      for (i <- 0 until bsize) {
        //println(t, i, oslen, osents(id + i).mkString(" "), next_x_t.mkString(" "))
        next_x_t.update(i, osents(id + i)(t))
      }
      // embed token
      val i_x_t = Expression.lookup(p_c, x_t)
      // run decoder step
      val i_y_t = decBuilder.addInput(i_x_t)
      // project from output dim to dictionary dimension
      val i_r_t = i_bias + i_R * i_y_t
      // Compute softmax and negative log
      val i_err = Expression.pickNegLogSoftmax(i_r_t, next_x_t)
      errs.add(i_err)
      x_t = next_x_t
    }

    // Sum loss over batch
    val i_nerr = Expression.sumBatches(Expression.sum(errs))
    i_nerr
  }

  // single sentence version of decode
  def decode(i_nc: Expression, osent: IntVector): Expression = {
    val osents = Seq(osent)
    decode(i_nc, osents, 0, 1)
  }

  def generate(insent: IntVector): IntVector = {
    generate(encode(insent), 2 * insent.size.toInt - 1)
  }

  // generate a sentence from an encoding
  def generate(i_nc: Expression, oslen: Int): IntVector = {
    val oein1 = new ExpressionVector()
    val oein2 = new ExpressionVector()
    val oein = new ExpressionVector()

    for (i <- 0 until numLayers) {
      oein1.add(Expression.pickrange(i_nc, i * hiddenDim, (i+1) * hiddenDim))
      oein2.add(Expression.tanh(oein1(i)))
    }

    for (i <- 0 until numLayers) oein.add(oein1(i))
    for (i <- 0 until numLayers) oein.add(oein2(i))

    decBuilder.newGraph()
    decBuilder.startNewSequence(oein)

    // decoder
    val i_R = Expression.parameter(p_R)

    val i_bias = Expression.parameter(p_bias)
    val osent = new IntVector()
    osent.add(EncoderDecoder.kSOS)

    var t = 0
    var done = false
    while (t < oslen && !done) {
      val i_x_t = Expression.lookup(p_c, osent(t))
      val i_y_t = decBuilder.addInput(i_x_t)
      val i_r_t = i_bias + i_R * i_y_t
      val i_ydist = Expression.softmax(i_r_t)
      val s = Utilities.sample(i_ydist.value.toVector)
      osent.add(s)
      if (s == EncoderDecoder.kEOS) done = true
      t += 1
    }

    osent
  }
}

object EncoderDecoder {
  var kSOS = 0
  var kEOS = 0

  var INPUT_VOCAB_SIZE = 0
  var OUTPUT_VOCAB_SIZE = 0

  val BATCH_SIZE = 1
  val DEV_BATCH_SIZE = 16
  val LAYERS = 1
  val INPUT_DIM = 2
  val HIDDEN_DIM = 4
  val BIDIRECTIONAL = false
  val NUM_EPOCHS = -1

  val userDir = System.getProperty("user.dir")

  val TRAIN_FILE = Paths.get(userDir, "../examples/cpp/example-data/train-hsm.txt").toString
  val DEV_FILE = Paths.get(userDir, "../examples/cpp/example-data/dev-hsm.txt").toString

  def main(args: Array[String]) {
    Initialize.initialize()

    val training = new scala.collection.mutable.ArrayBuffer[IntVector]
    val dev = new scala.collection.mutable.ArrayBuffer[IntVector]

    val d = new WordDict
    kSOS = d.convert("<s>")
    kEOS = d.convert("</s>")

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

    val model = new ParameterCollection()
    val adam = new AdamTrainer(model, 0.001f, 0.9f, 0.999f, 1e-8f)
    adam.clipThreshold = adam.clipThreshold * BATCH_SIZE

    // create model
    val lm = new EncoderDecoder(model, LAYERS, INPUT_DIM, HIDDEN_DIM, BIDIRECTIONAL)

    // number of batches in training set
    val numBatches = training.size / BATCH_SIZE - 1
    println(s"numBatches ${numBatches}")

    // Random indexing
    val order = new IntVector(0 until numBatches)

    var first = true
    var epoch = 0

    ComputationGraph.renew()

    // run for the given number of epochs (or forever if NUM_EPOCHS is negaive)
    while (epoch < NUM_EPOCHS || NUM_EPOCHS < 0) {
      // update the optimizer
      if (first) { first = false } else { adam.updateEpoch() }
      // reshuffle the dataset
      Utilities.shuffle(order)
      // initialize loss and number of chars per word
      var loss = 0.0
      val chars = new IntPointer
      chars.set(0)

      for (si <- 0 until numBatches) {
        // build graph for this instance
        ComputationGraph.clear()
        // compute batch start id and size
        val id = order(si) * BATCH_SIZE
        val bsize = math.min(training.size - id, BATCH_SIZE)
        // encode the batch
        val encoding = lm.encode(training, id, bsize, chars)
        // decode and get error
        val loss_expr = lm.decode(encoding, training, id, bsize)
        // get scalar error for monitoring
        loss += ComputationGraph.forward(loss_expr).toFloat
        // compute gradient with backward pass
        ComputationGraph.backward(loss_expr)
        // update parameters
        adam.update()
        // Print progress every tenth of the dataset
        if ((si + 1) % (numBatches / 10) == 0 || si == numBatches - 1) {
          // adam.status()
          val lc = loss / chars.value
          println(s"${si} E = ${lc} ppl=${math.exp(lc)}")
          // reinitialize loss
          loss = 0.0
          chars.set(0)
        }
      }

      // show score on dev data
      var dloss = 0.0
      val dchars = new IntPointer
      dchars.set(0)

      for (i <- 0 until (dev.size / DEV_BATCH_SIZE)) {
        // clear graph
        ComputationGraph.clear()

        // compute batch start id and size
        val id = i * DEV_BATCH_SIZE
        val bsize = math.min(dev.size - id, DEV_BATCH_SIZE)
        // Encode
        val encoding = lm.encode(dev, id, bsize, dchars)
        // Decode and get loss
        val loss_expr = lm.decode(encoding, dev, id, bsize)
        // Count loss
        dloss += ComputationGraph.forward(loss_expr).toFloat
      }
      val dlc = dloss / dchars.value
      println(s"DEV [epoch=${epoch}] E = ${dlc} ppl=${math.exp(dlc)}")

      // sample some examples because it's cool
      println("------------------------")

      for (_ <- 1 to 5) {
        // select a random sentence from the dev set
        val idx = scala.util.Random.nextInt(dev.size)
        val sent = dev(idx)

        val originalSentence = sent.map(d.convert).mkString(" ")
        println(s"original sentence $idx: ${originalSentence}")

        val sampled = lm.generate(sent)
        val sampledSentence = sampled.map(d.convert).mkString(" ")
        println(s"sampled sentence: ${sampledSentence}")
      }
      println("----------------------")

      // increment epoch
      epoch += 1
    }

  }
}
