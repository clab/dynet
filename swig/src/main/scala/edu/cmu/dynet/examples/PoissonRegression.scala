package edu.cmu.dynet.examples

import edu.cmu.dynet._
import edu.cmu.dynet.dynet_swig._

import scala.language.implicitConversions

import java.nio.file.Paths

object PoissonRegression {
  import DynetScalaHelpers._

  val LAYERS = 2
  val INPUT_DIM = 16
  val HIDDEN_DIM = 32
  var VOCAB_SIZE = 0

  class RNNLengthPredictor(model: Model) {
    val builder = new LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
    val p_c = model.add_lookup_parameters(VOCAB_SIZE, dim(INPUT_DIM))
    val p_R = model.add_parameters(dim(1, HIDDEN_DIM))
    val p_bias = model.add_parameters(dim(1))

    def buildLMGraph(sent: IntVector,
                     len: Int,
                     cg: ComputationGraph,
                     flag: Boolean=false): Expression = {
      val slen = (sent.size - 1).toInt
      builder.new_graph(cg)
      builder.start_new_sequence()

      val R = parameter(cg, p_R)
      val bias = parameter(cg, p_bias)

      for (t <- 0 until slen) {
        val i_x_t = lookup(cg, p_c, sent.get(t))
        builder.add_input(i_x_t)
      }

      val pred = affine_transform(Seq(bias, R, builder.back))

      if (flag) {
        val x = math.exp(cg.incremental_forward(pred).toFloat)
        println(s"PRED = ${x} TRUE = ${len} DIFF = ${x - len}")
      }

      poisson_loss(pred, len)
    }
  }

  def shuffle(vs: IntVector): Unit = {
    val values = for (i <- 0 until vs.size.toInt) yield vs.get(i)
    scala.util.Random.shuffle(values)
    values.zipWithIndex.foreach { case (v, i) => vs.set(i, v) }
  }

  def main(args: Array[String]) {
    myInitialize()

    val userDir = System.getProperty("user.dir")

    val CORPUS_FILE = Paths.get(userDir, "../examples/cpp/example-data/train-poi.txt").toString
    val DEV_FILE = Paths.get(userDir, "../examples/cpp/example-data/dev-poi.txt").toString

    val d = new Dict()
    val kSOS = d.convert("<s>")
    val kEOS = d.convert("</s>")

    val training = new scala.collection.mutable.ArrayBuffer[(IntVector, Int)]
    val dev = new scala.collection.mutable.ArrayBuffer[(IntVector, Int)]

    var tlc = 0
    var ttoks = 0

    {
      val td = new Dict()
      for (line <- scala.io.Source.fromFile(CORPUS_FILE).getLines) {
        tlc += 1
        val x = new IntVector()
        val ty = new IntVector()

        read_sentence_pair(line, x, d, ty, td)
        assert(ty.size == 1)
        val v = td.convert(ty.get(0))

        for (c <- v.toCharArray) {
          assert(c >= '0' && c <= '9')
        }

        val y = v.toInt

        training.append((x, y))

        ttoks += x.size().toInt

        if (x.get(0) != kSOS && x.get(x.size.toInt - 1) != kEOS) {
          throw new RuntimeException("bad sentence")
        }
      }
      println(s"${tlc} lines ${ttoks} tokens ${d.size} types")
    }

    d.freeze()
    VOCAB_SIZE = d.size.toInt

    var dlc = 0
    var dtoks = 0

    {
      val td = new Dict
      for (line <- scala.io.Source.fromFile(DEV_FILE).getLines) {
        dlc += 1
        val x = new IntVector()
        val ty = new IntVector()

        read_sentence_pair(line, x, d, ty, td)
        assert(ty.size == 1)
        val v = td.convert(ty.get(0))

        for (c <- v.toCharArray) {
          assert(c >= '0' && c <= '9')
        }

        val y = v.toInt

        dev.append((x, y))

        dtoks += x.size().toInt

        if (x.get(0) != kSOS && x.get(x.size.toInt - 1) != kEOS) {
          throw new RuntimeException("bad sentence")
        }
      }
      println(s"${dlc} lines ${dtoks} tokens")
    }

    var best = Float.MaxValue

    val model = new Model()
    val sgd = new SimpleSGDTrainer(model)

    val lm = new RNNLengthPredictor(model)

    val report_every_i = 50
    val dev_every_i_reports = 20
    var si = training.size
    val order = new IntVector((0 until training.size))

    var first = true
    var report = 0
    var lines = 0

    val cg = new ComputationGraph()

    while (true) {
      var loss = 0.0f
      var chars = 0
      for (i <- 0 until report_every_i) {
        if (si == training.size) {
          si = 0
          if (first) {
            first = false
          } else {
            sgd.update_epoch()
          }

          shuffle(order)
        }

        // build graph for this instance

        // the cg.clear is IMPORTANT!
        cg.clear()

        val sent = training(order.get(si))

        si += 1

        val loss_expr = lm.buildLMGraph(sent._1, sent._2, cg)
        loss += cg.forward(loss_expr).toFloat
        cg.backward(loss_expr)
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

        for (sent <- dev) {
          val loss_expr = lm.buildLMGraph(sent._1, sent._2, cg, true)
          dloss += cg.forward(loss_expr).toFloat
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
