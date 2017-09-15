package edu.cmu.dynet.examples

import edu.cmu.dynet._

import scala.language.implicitConversions

import java.nio.file.Paths

object PoissonRegression {

  val LAYERS = 2
  val INPUT_DIM = 16
  val HIDDEN_DIM = 32
  var VOCAB_SIZE = 0

  class RNNLengthPredictor(model: ParameterCollection) {
    val builder = new LstmBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    val p_c = model.addLookupParameters(VOCAB_SIZE, Dim(INPUT_DIM))
    val p_R = model.addParameters(Dim(1, HIDDEN_DIM))
    val p_bias = model.addParameters(Dim(1))

    def buildLMGraph(sent: IntVector,
                     len: Int,
                     flag: Boolean=false): Expression = {
      val slen = (sent.size - 1)
      builder.newGraph()
      builder.startNewSequence()

      val R = Expression.parameter(p_R)
      val bias = Expression.parameter(p_bias)

      for (t <- 0 until slen) {
        val i_x_t = Expression.lookup(p_c, sent(t))
        builder.addInput(i_x_t)
      }

      val ev = new ExpressionVector(Seq(bias, R, builder.back))
      val pred = Expression.affineTransform(ev)

      if (flag) {
        val x = math.exp(ComputationGraph.incrementalForward(pred).toFloat)
        println(s"PRED = ${x} TRUE = ${len} DIFF = ${x - len}")
      }

      Expression.poissonLoss(pred, len)
    }
  }

  def main(args: Array[String]) {
    Initialize.initialize()

    val userDir = System.getProperty("user.dir")

    val CORPUS_FILE = Paths.get(userDir, "../examples/cpp/example-data/train-poi.txt").toString
    val DEV_FILE = Paths.get(userDir, "../examples/cpp/example-data/dev-poi.txt").toString

    val d = new WordDict()
    val kSOS = d.convert("<s>")
    val kEOS = d.convert("</s>")

    val training = new scala.collection.mutable.ArrayBuffer[(IntVector, Int)]
    val dev = new scala.collection.mutable.ArrayBuffer[(IntVector, Int)]

    var tlc = 0
    var ttoks = 0

    {
      val td = new WordDict()
      for (line <- scala.io.Source.fromFile(CORPUS_FILE).getLines) {
        tlc += 1

        val (x, y) = WordDict.read_sentence_pair(line, d, td)

        training.append((x, y))

        ttoks += x.size

        if (x(0) != kSOS && x.last != kEOS) {
          throw new RuntimeException("bad sentence")
        }
      }
      println(s"${tlc} lines ${ttoks} tokens ${d.size} types")
    }

    d.freeze()
    VOCAB_SIZE = d.size

    var dlc = 0
    var dtoks = 0

    {
      val td = new WordDict
      for (line <- scala.io.Source.fromFile(DEV_FILE).getLines) {
        dlc += 1

        val (x, y) = WordDict.read_sentence_pair(line, d, td)

        dev.append((x, y))

        dtoks += x.size

        if (x(0) != kSOS && x.last != kEOS) {
          throw new RuntimeException("bad sentence")
        }
      }
      println(s"${dlc} lines ${dtoks} tokens")
    }

    var best = Float.MaxValue

    val model = new ParameterCollection()
    val sgd = new SimpleSGDTrainer(model)

    val lm = new RNNLengthPredictor(model)

    val report_every_i = 50
    val dev_every_i_reports = 20
    var si = training.size
    val order = new IntVector((0 until training.size))

    var first = true
    var report = 0
    var lines = 0

    ComputationGraph.renew()

    while (true) {
      var loss = 0.0f
      var chars = 0
      for (i <- 0 until report_every_i) {
        if (si == training.size) {
          si = 0
          if (first) {
            first = false
          } else {
            sgd.updateEpoch()
          }

          Utilities.shuffle(order)
        }

        // build graph for this instance

        // the cg.clear is IMPORTANT!
        ComputationGraph.clear()

        val (tokens, count) = training(order(si))

        si += 1

        val loss_expr = lm.buildLMGraph(tokens, count)
        loss += ComputationGraph.forward(loss_expr).toFloat
        ComputationGraph.backward(loss_expr)
        sgd.update()
        lines += 1
        chars += 1
      }
      sgd.status()
      println(s"E = ${loss / chars.toFloat} ppl = ${math.exp(loss / chars.toFloat)}")

      report += 1
      if (report % dev_every_i_reports == 0) {
        var dloss = 0.0f
        var dchars = 0

        for ((tokens, count) <- dev) {
          val loss_expr = lm.buildLMGraph(tokens, count, true)
          dloss += ComputationGraph.forward(loss_expr).toFloat
          dchars += 1
        }

        if (dloss < best) {
          best = dloss
          println(s"new best: ${best}")
        }

        println(s"DEV [epoch = ${lines / training.size.toFloat}] " +
                s"E = ${dloss / dchars.toFloat} " +
                s"ppl = ${math.exp(dloss / dchars.toFloat)}")
      }
    }
  }
}
