package edu.cmu.dynet.examples

import edu.cmu.dynet.IntVector

// Stripped-down replacement for dynet/dict.h
class WordDict {
  val mapping = new scala.collection.mutable.HashMap[String, Int]
  val words = new scala.collection.mutable.ArrayBuffer[String]

  var frozen = false
  var mapUnk = false
  var unkId = -1

  def size(): Int = words.size
  def freeze(): Unit = { frozen = true }
  def is_frozen(): Boolean = frozen

  def contains(word: String): Boolean = words.contains(word)

  def convert(word: String): Int = mapping.get(word) match {
    case Some(i) => i
    case None if frozen && mapUnk => unkId
    case None if frozen => throw new RuntimeException("unknown word in frozen dict")
    case None => {
      val index = mapping.size
      mapping.put(word, index)
      words.append(word)
      index
    }
  }

  def convert(i: Int): String = words(i)

  def set_unk(s: String) = {
    if (!frozen) throw new RuntimeException("called set_unk on unfrozen dict")
    if (mapUnk) throw new RuntimeException("called set_unk more than once")

    frozen = false
    unkId = convert(s)
    frozen = true
    mapUnk = true
  }

  def getUnkId(): Int = unkId
}

object WordDict {
  def read_sentence(line: String, sd: WordDict): IntVector = {
    new IntVector(line.split(" ").map(sd.convert).toSeq)
  }

  def read_sentence_pair(line: String, sd: WordDict, td: WordDict): (IntVector, Int) = {
    val Array(before, after) = line.split(""" \|\|\| """)
    val tokens = read_sentence(before, sd)
    val count = td.convert(read_sentence(after, td)(0)).toInt
    (tokens, count)
  }
}