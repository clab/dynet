package edu.cmu.dynet

class Expression private[dynet](private[dynet] val expr: internal.Expression) {
  // give it the current version
  val version = ComputationGraph.version

  def value(): Tensor = new Tensor(expr.value)
  def dim(): Dim = new Dim(expr.dim)

  private[dynet] def ensureFresh(): Unit = {
    if (version != ComputationGraph.version) throw new RuntimeException("stale expression")
  }

  // Sugar for arithmetic
  def +(e2: Expression): Expression = Expression.exprPlus(this, e2)
  def *(e2: Expression): Expression = Expression.exprTimes(this, e2)
  def -(e2: Expression): Expression = Expression.exprMinus(this, e2)
  def +(r: Float): Expression = Expression.exprPlus(this, r)
  def *(r: Float): Expression = Expression.exprTimes(this, r)
  def -(r: Float): Expression = Expression.exprMinus(this, r)
  def /(r: Float): Expression = Expression.exprDivide(this, r)
  def unary_-(): Expression = Expression.exprMinus(this)
}

object Expression {
  import edu.cmu.dynet.internal.{dynet_swig => dn}
  import ImplicitConversions._

  // private helper function for wrapping methods that get expressions from computation graph
  private def makeExpr(f: internal.ComputationGraph => internal.Expression): Expression = {
    val version = ComputationGraph.version
    val expr = f(ComputationGraph.cg)
    new Expression(expr)
  }

  def input(s: Float): Expression = makeExpr(cg => dn.input(ComputationGraph.cg, s))
  def input(fp: FloatPointer): Expression = makeExpr(cg => dn.input(ComputationGraph.cg, fp))
  def input(d: Dim, pdata: FloatVector): Expression = makeExpr(cg => dn.input(cg, d, pdata))
  //def input(d: Dim, )

  def parameter(p: Parameter): Expression = makeExpr(cg => dn.parameter(cg, p))
  def constParameter(p: Parameter): Expression = makeExpr(cg => dn.const_parameter(cg, p))

  def lookup(p: LookupParameter, index: Long) = makeExpr(cg => dn.lookup(cg, p, index))
  def lookup(p: LookupParameter, pindex: UnsignedPointer) = makeExpr(cg => dn.lookup(cg, p, pindex))
  def constLookup(p: LookupParameter, index: Long) = makeExpr(cg => dn.lookup(cg, p, index))
  // def constLookup
  def lookup(p: LookupParameter, indices: UnsignedVector) = makeExpr(cg => dn.lookup(cg, p, indices))


  def zeroes(d: Dim) = makeExpr(cg => dn.zeroes(cg, d))
  def randomNormal(d: Dim) = makeExpr(cg => dn.random_normal(cg, d))
  def randomBernoulli(d: Dim, p: Float, scale: Float = 1.0f) = makeExpr(
    cg => dn.random_bernoulli(cg, d, p, scale))
  def randomUniform(d: Dim, left: Float, right: Float) = makeExpr(
    cg => dn.random_uniform(cg, d, left, right))

  /* ARITHMETIC OPERATIONS */

  private type BinaryTransform = (internal.Expression, internal.Expression) => internal.Expression
  private def binary(e1: Expression, e2: Expression, combiner: BinaryTransform) = {
    e1.ensureFresh()
    e2.ensureFresh()
    val expr = combiner(e1.expr, e2.expr)
    new Expression(expr)
  }

  private type UnaryTransform = internal.Expression => internal.Expression
  private def unary(e: Expression, transformer: UnaryTransform) = {
    e.ensureFresh()
    new Expression(transformer(e))
  }

  def exprMinus(e: Expression): Expression = unary(e, dn.exprMinus)
  def exprPlus(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.exprPlus)
  def exprPlus(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprPlus(e1, x))
  def exprPlus(x: Float, e2: Expression): Expression = exprPlus(e2, x)
  def exprMinus(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.exprMinus)
  def exprMinus(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprMinus(e1, x))
  def exprMinus(x: Float, e2: Expression): Expression = unary(e2, e2 => dn.exprMinus(x, e2))
  def exprTimes(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.exprTimes)
  def exprTimes(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprTimes(e1, x))
  def exprTimes(x: Float, e2: Expression): Expression = unary(e2, e2 => dn.exprTimes(x, e2))
  def exprDivide(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprDivide(e1, x))

  private type VectorTransform = internal.ExpressionVector => internal.Expression
  private def vectory(v: ExpressionVector, transformer: VectorTransform) = {
    v.ensureFresh()
    new Expression(transformer(v.vector))
  }

  def affineTransform(ev: ExpressionVector): Expression = vectory(ev, dn.affine_transform)
  def sum(ev: ExpressionVector): Expression = vectory(ev, dn.sum)
  def average(ev: ExpressionVector): Expression = vectory(ev, dn.average)

  def sqrt(e: Expression): Expression = unary(e, dn.sqrt)
  def erf(e: Expression): Expression = unary(e, dn.erf)
  def tanh(e: Expression): Expression = unary(e, dn.tanh)
  def exp(e: Expression): Expression = unary(e, dn.exp)
  def square(e: Expression): Expression = unary(e, dn.square)
  def cube(e: Expression): Expression = unary(e, dn.cube)
  def lgamma(e: Expression): Expression = unary(e, dn.lgamma)
  def log(e: Expression): Expression = unary(e, dn.log)
  def logistic(e: Expression): Expression = unary(e, dn.logistic)
  def rectify(e: Expression): Expression = unary(e, dn.rectify)
  def softsign(e: Expression): Expression = unary(e, dn.softsign)
  def pow(x: Expression, y: Expression): Expression = binary(x, y, dn.pow)

  def min(x: Expression, y: Expression): Expression = binary(x, y, dn.min)

  def max(x: Expression, y: Expression): Expression = binary(x, y, dn.max)
  def max(v: ExpressionVector): Expression = vectory(v, dn.max)
  def dotProduct(x: Expression, y: Expression): Expression = binary(x, y, dn.dot_product)
  def cmult(x: Expression, y: Expression): Expression = binary(x, y, dn.cmult)
  def cdiv(x: Expression, y: Expression): Expression = binary(x, y, dn.cdiv)
  def colwiseAdd(x: Expression, bias: Expression): Expression = binary(x, bias, dn.colwise_add)

  /* PROBABILITY / LOSS OPERATIONS */

  def softmax(e: Expression): Expression = unary(e, dn.softmax)
  def logSoftmax(e: Expression): Expression = unary(e, dn.log_softmax)
  // def logSoftmax

  def logSumExp(v: ExpressionVector): Expression = vectory(v, dn.logsumexp)

  def pickNegLogSoftmax(e: Expression, v: Long): Expression = unary(e, e => dn.pickneglogsoftmax(e, v))
  def pickNegLogSoftmax(e: Expression, v: UnsignedPointer): Expression =
    unary(e, e => dn.pickneglogsoftmax(e, v))
  def pickNegLogSoftmax(e: Expression, v: UnsignedVector): Expression =
    unary(e, e => dn.pickneglogsoftmax(e, v))

  def hinge(e: Expression, index: Long, m: Float = 1.0f): Expression = unary(e, e => dn.hinge(e, index, m))
  // others

  def sparsemax(e: Expression): Expression = unary(e, dn.sparsemax)
  // def sparsemaxLoss

  def squaredNorm(e: Expression): Expression = unary(e, dn.squared_norm)
  def squaredDistance(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.squared_distance)
  def l1Distance(x: Expression, y: Expression): Expression = binary(x, y, dn.l1_distance)
  def huberDistance(x: Expression, y: Expression, c: Float = 1.345f) = {
    binary(x, y, (x, y) => dn.huber_distance(x, y, c))
  }
  def binaryLogLoss(x: Expression, y: Expression): Expression = binary(x, y, dn.binary_log_loss)
  def pairwiseRankLoss(x: Expression, y: Expression, m: Float = 1.0f) =
    binary(x, y, (x, y) => dn.pairwise_rank_loss(x, y, m))
  def poissonLoss(x: Expression, y: Long): Expression = unary(x, x => dn.poisson_loss(x, y))
  // others

  /* FLOW / SHAPING OPERATIONS */

  def noBackProp(x: Expression): Expression = unary(x, dn.nobackprop)
  def reshape(x: Expression, d: Dim): Expression = unary(x, x => dn.reshape(x, d))
  def transpose(x: Expression): Expression = unary(x, dn.transpose)
  //def selectRows()
  //def selectCols()
  def sumBatches(x: Expression): Expression = unary(x, dn.sum_batches)

  // def pick
  def pickrange(x: Expression, v: Long, u: Long) = unary(x, x => dn.pickrange(x, v, u))

  def concatenateCols(v: ExpressionVector): Expression = vectory(v, dn.concatenate_cols)
  def concatenate(v: ExpressionVector): Expression = vectory(v, dn.concatenate)

  /* NOISE OPERATIONS */

  def noise(x: Expression, stddev: Float): Expression = unary(x, x => dn.noise(x, stddev))
  def dropout(x: Expression, p: Float): Expression = unary(x, x => dn.dropout(x, p))
  def blockDropout(x: Expression, p: Float): Expression = unary(x, x => dn.block_dropout(x, p))

  /* CONVOLUTION OPERATIONS */

  /* TENSOR OPERATIONS */

  /* LINEAR ALGEBRA OPERATIONS */
}

