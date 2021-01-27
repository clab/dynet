package edu.cmu.dynet

/** Represents an expression on the computation graph. Can only be constructed using the
  * functions contained in the companion object.
  */
class Expression private[dynet](
  private[dynet] val expr: internal.Expression,
  // Expressions sometimes rely on things (e.g. wrapped C++ vectors) that get deleted when the JVM
  // garbage collector runs. By explicitly grabbing references to them, we can prevent this
  // premature garbage collection.
  val reference: AnyRef = null
) {
  // Give it the current version
  val version = ComputationGraph.version

  /** Get the tensor value of this expression */
  def value(): Tensor = new Tensor(expr.value)

  /** Get the tensor dimension of this expression */
  def dim(): Dim = new Dim(expr.dim)

  /** Make sure that this expression is the latest version */
  private[dynet] def ensureFresh(): Unit = {
    if (version != ComputationGraph.version) throw new RuntimeException("stale expression")
  }

  // Sugar for doing expression arithmetic
  def +(e2: Expression): Expression = Expression.exprPlus(this, e2)
  def *(e2: Expression): Expression = Expression.exprTimes(this, e2)
  def -(e2: Expression): Expression = Expression.exprMinus(this, e2)
  def +(r: Float): Expression = Expression.exprPlus(this, r)
  def *(r: Float): Expression = Expression.exprTimes(this, r)
  def -(r: Float): Expression = Expression.exprMinus(this, r)
  def /(r: Float): Expression = Expression.exprDivide(this, r)
  def unary_-(): Expression = Expression.exprMinus(this)

  def debugString(): String = s"(Expression: ${dim.debugString} ${value.toSeq})"
}

/** Contains methods for creating [[edu.cmu.dynet.Expression]]s. There are several ways to create
  *  expressions:
  *
  *  * from explicit values (e.g. `input`)
  *  * randomly (e.g. `randomNormal`)
  *  * from [[edu.cmu.dynet.ParameterCollection]] parameters (e.g. `parameter`)
  *  * from other expressions (e.g. `softmax` and `pow`)
  */
object Expression {
  import edu.cmu.dynet.internal.{dynet_swig => dn}

  /** Private helper function for wrapping methods that get expressions from the computation
    * graph */
  private def makeExpr(
    f: internal.ComputationGraph => internal.Expression,
    reference: AnyRef = null
  ): Expression = {
    val version = ComputationGraph.version
    val expr = f(ComputationGraph.cg)
    new Expression(expr, reference)
  }

  def input(s: Float): Expression = makeExpr(cg => dn.input(ComputationGraph.cg, s))
  def input(fp: FloatPointer): Expression =
    makeExpr(cg => dn.input(ComputationGraph.cg, fp.floatp), fp)
  def input(d: Dim, pdata: FloatVector): Expression =
    makeExpr(cg => dn.input(cg, d.dim, pdata.vector), Seq(d, pdata))
  def input(d: Dim, ids: UnsignedVector, data: FloatVector, defdata: Float = 0f) =
    makeExpr(cg => dn.input(cg, d.dim, ids.vector, data.vector, defdata), Seq(d, ids, data))

  def parameter(p: Parameter): Expression = makeExpr(cg => dn.parameter(cg, p.parameter), p)
  def parameter(lp: LookupParameter): Expression = makeExpr(cg => dn.parameter(cg, lp.lookupParameter), lp)
  def constParameter(p: Parameter): Expression =
    makeExpr(cg => dn.const_parameter(cg, p.parameter), p)
  def constParameter(lp: LookupParameter): Expression =
    makeExpr(cg => dn.const_parameter(cg, lp.lookupParameter), lp)

  def lookup(p: LookupParameter, index: Long) =
    makeExpr(cg => dn.lookup(cg, p.lookupParameter, index), p)
  def lookup(p: LookupParameter, pindex: UnsignedPointer) =
    makeExpr(cg => dn.lookup(cg, p.lookupParameter, pindex.uintp), Seq(p, pindex))
  def constLookup(p: LookupParameter, index: Long) =
    makeExpr(cg => dn.const_lookup(cg, p.lookupParameter, index), p)
  def constLookup(p: LookupParameter, pindex: UnsignedPointer) =
    makeExpr(cg => dn.const_lookup(cg, p.lookupParameter, pindex.uintp), Seq(p, pindex))
  def lookup(p: LookupParameter, indices: UnsignedVector) =
    makeExpr(cg => dn.lookup(cg, p.lookupParameter, indices.vector), Seq(p, indices))
  def constLookup(p: LookupParameter, indices: UnsignedVector) =
    makeExpr(cg => dn.const_lookup(cg, p.lookupParameter, indices.vector), Seq(p, indices))

  def zeros(d: Dim) = makeExpr(cg => dn.zeros(cg, d.dim), d)
  def zeroes(d: Dim) = makeExpr(cg => dn.zeros(cg, d.dim), d)
  def ones(d: Dim) = makeExpr(cg => dn.ones(cg, d.dim), d)
  def constant(d: Dim, v: Float) = makeExpr(cg => dn.constant(cg, d.dim, v), d)
  def randomNormal(d: Dim) = makeExpr(cg => dn.random_normal(cg, d.dim), d)
  def randomBernoulli(d: Dim, p: Float, scale: Float = 1.0f) = makeExpr(
    cg => dn.random_bernoulli(cg, d.dim, p, scale), d)
  def randomUniform(d: Dim, left: Float, right: Float) = makeExpr(
    cg => dn.random_uniform(cg, d.dim, left, right), d)
  def randomGumbel(d: Dim, mu: Float, beta: Float) = makeExpr(
    cg => dn.random_gumbel(cg, d.dim, mu, beta), d)

  /* ARITHMETIC OPERATIONS */

  private type BinaryTransform = (internal.Expression, internal.Expression) => internal.Expression
  private def binary(e1: Expression, e2: Expression, combiner: BinaryTransform) = {
    e1.ensureFresh()
    e2.ensureFresh()
    val expr = combiner(e1.expr, e2.expr)
    // Specify e1 and e2 as references so they can't get prematurely garbage collected.
    new Expression(expr, Seq(e1, e2))
  }

  private type UnaryTransform = internal.Expression => internal.Expression
  private def unary(e: Expression, transformer: UnaryTransform) = {
    e.ensureFresh()
    // Specify e as reference so it can't get prematurely garbage collected.
    new Expression(transformer(e.expr), e)
  }

  def exprMinus(e: Expression): Expression = unary(e, dn.exprMinus)
  def exprPlus(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.exprPlus)
  def exprPlus(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprPlus(e1, x))
  def exprPlus(x: Float, e2: Expression): Expression = unary(e2, e2 => dn.exprPlus(x, e2))
  def exprMinus(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.exprMinus)
  def exprMinus(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprMinus(e1, x))
  def exprMinus(x: Float, e2: Expression): Expression = unary(e2, e2 => dn.exprMinus(x, e2))
  def exprTimes(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.exprTimes)
  def exprTimes(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprTimes(e1, x))
  def exprTimes(x: Float, e2: Expression): Expression = unary(e2, e2 => dn.exprTimes(x, e2))
  def exprDivide(e1: Expression, x: Float): Expression = unary(e1, e1 => dn.exprDivide(e1, x))

  private type VectorTransform = internal.ExpressionVector => internal.Expression
  private def vectory(v: ExpressionVector, transformer: VectorTransform) = {
    // DyNet segfaults if we pass a zero-length vector.
    // This check results in a nicer error message.
    assert(v.nonEmpty, "Operation requires > 0 expression arguments")
    v.ensureFresh()
    // Specify v as reference so it can't get prematurely garbage collected.
    new Expression(transformer(v.vector), v)
  }

  def affineTransform(ev: ExpressionVector): Expression = vectory(ev, dn.affine_transform)
  def affineTransform(exprs: Expression*): Expression = affineTransform(new ExpressionVector(exprs))

  def sum(ev: ExpressionVector): Expression = vectory(ev, dn.sum)
  def sum(exprs: Expression*): Expression = sum(new ExpressionVector(exprs))

  def sumElems(e: Expression): Expression = unary(e, dn.sum_elems)
  def momentElems(e: Expression, r: Long) = unary(e, e => dn.moment_elems(e, r))
  def meanElems(e: Expression): Expression = unary(e, dn.mean_elems)
  def stdElems(e: Expression): Expression = unary(e, dn.std_elems)

  def average(ev: ExpressionVector): Expression = vectory(ev, dn.average)
  def average(exprs: Expression*): Expression = average(new ExpressionVector(exprs))

  def sqrt(e: Expression): Expression = unary(e, dn.sqrt)
  def abs(e: Expression): Expression = unary(e, dn.abs)
  def erf(e: Expression): Expression = unary(e, dn.erf)
  def tanh(e: Expression): Expression = unary(e, dn.tanh)
  def exp(e: Expression): Expression = unary(e, dn.exp)
  def square(e: Expression): Expression = unary(e, dn.square)
  def cube(e: Expression): Expression = unary(e, dn.cube)
  def lgamma(e: Expression): Expression = unary(e, dn.lgamma)
  def log(e: Expression): Expression = unary(e, dn.log)
  def logistic(e: Expression): Expression = unary(e, dn.logistic)
  def rectify(e: Expression): Expression = unary(e, dn.rectify)
  def elu(e: Expression): Expression = unary(e, dn.elu)
  def selu(e: Expression): Expression = unary(e, dn.selu)
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

  def softmax(e: Expression, d: Long = 0L): Expression = unary(e, e => dn.softmax(e, d))
  def logSoftmax(e: Expression): Expression = unary(e, dn.log_softmax)
  def logSoftmax(e: Expression, restriction: UnsignedVector) =
    unary(e, e => dn.log_softmax(e, restriction.vector))

  def logSumExp(v: ExpressionVector): Expression = vectory(v, dn.logsumexp)

  def pickNegLogSoftmax(e: Expression, v: Long): Expression = unary(e, e => dn.pickneglogsoftmax(e, v))
  def pickNegLogSoftmax(e: Expression, v: UnsignedPointer): Expression =
    unary(e, e => dn.pickneglogsoftmax(e, v.uintp))
  def pickNegLogSoftmax(e: Expression, v: UnsignedVector): Expression =
    unary(e, e => dn.pickneglogsoftmax(e, v.vector))

  def hinge(e: Expression, index: Long, m: Float = 1.0f): Expression = unary(e, e => dn.hinge(e, index, m))
  def hinge(e: Expression, index: UnsignedPointer, m: Float): Expression =
    unary(e, e => dn.hinge(e, index.uintp, m))
  def hinge(e: Expression, indices: UnsignedVector, m: Float): Expression =
    unary(e, e => dn.hinge(e, indices.vector, m))

  def hinge(e: Expression, index: UnsignedPointer): Expression =
    unary(e, e => dn.hinge(e, index.uintp, 1.0f))
  def hinge(e: Expression, indices: UnsignedVector): Expression =
    unary(e, e => dn.hinge(e, indices.vector, 1.0f))

  def hingeDim(e: Expression, indices: UnsignedVector, d: Long = 0L, m: Float = 1.0f): Expression =
    unary(e, e => dn.hinge_dim(e, indices.vector, d, m))
  def hingeDimBatch(e: Expression, indices: UnsignedVectorVector, d: Long = 0L, m: Float = 1.0f): Expression =
    unary(e, e => dn.hinge_dim(e, indices.vector, d, m))

  def sparsemax(e: Expression): Expression = unary(e, dn.sparsemax)
  def sparsemaxLoss(e: Expression, targetSupport: UnsignedVector): Expression =
    unary(e, e => dn.sparsemax_loss(e, targetSupport.vector))

  def squaredNorm(e: Expression): Expression = unary(e, dn.squared_norm)
  def l2Norm(e: Expression): Expression = unary(e, dn.l2_norm)
  def squaredDistance(e1: Expression, e2: Expression): Expression = binary(e1, e2, dn.squared_distance)
  def l1Distance(x: Expression, y: Expression): Expression = binary(x, y, dn.l1_distance)
  def huberDistance(x: Expression, y: Expression, c: Float = 1.345f) = {
    binary(x, y, (x, y) => dn.huber_distance(x, y, c))
  }
  def binaryLogLoss(x: Expression, y: Expression): Expression = binary(x, y, dn.binary_log_loss)
  def pairwiseRankLoss(x: Expression, y: Expression, m: Float = 1.0f) =
    binary(x, y, (x, y) => dn.pairwise_rank_loss(x, y, m))
  def poissonLoss(x: Expression, y: Long): Expression = unary(x, x => dn.poisson_loss(x, y))
  def poissonLoss(x: Expression, y: UnsignedPointer): Expression =
    unary(x, x => dn.poisson_loss(x, y.uintp))

  /* FLOW / SHAPING OPERATIONS */

  def noBackProp(x: Expression): Expression = unary(x, dn.nobackprop)
  def flipGradient(x: Expression): Expression = unary(x, dn.flip_gradient)
  def reshape(x: Expression, d: Dim): Expression = unary(x, x => dn.reshape(x, d.dim))
  def transpose(x: Expression): Expression = unary(x, dn.transpose)
  def selectRows(x: Expression, rows: UnsignedVector): Expression =
    unary(x, x => dn.select_rows(x, rows.vector))
  def selectCols(x: Expression, rows: UnsignedVector): Expression =
    unary(x, x => dn.select_cols(x, rows.vector))
  def sumBatches(x: Expression): Expression = unary(x, dn.sum_batches)
  def momentBatches(x: Expression, r: Long): Expression = unary(x, x => dn.moment_batches(x, r))
  def stdBatches(x: Expression): Expression = unary(x, dn.std_batches)
  def momentDim(x: Expression, v: UnsignedVector, r: Long, b: Boolean = false, n: Long = 0L): Expression =
    unary(x, x => dn.moment_dim(x, v.vector, r, b, n))
  def meanDim(x: Expression, v: UnsignedVector, b: Boolean = false, n: Long = 0L): Expression =
    unary(x, x => dn.mean_dim(x, v.vector, b, n))
  def stdDim(x: Expression, v: UnsignedVector, b: Boolean = false, n: Long = 0L): Expression =
    unary(x, x => dn.std_dim(x, v.vector, b, n))

  def pick(x: Expression, v: Long, d: Long = 0l): Expression = unary(x, x => dn.pick(x, v, d))
  def pick(x: Expression, v: UnsignedVector, d: Long): Expression =
    unary(x, x => dn.pick(x, v.vector, d))
  def pick(x: Expression, v: UnsignedPointer, d: Long): Expression =
    unary(x, x => dn.pick(x, v.uintp, d))
  def pickrange(x: Expression, v: Long, u: Long, d: Long = 0l): Expression =
    unary(x, x => dn.pick_range(x, v, u, d))
  def pickBatchElem(x: Expression, v: Long): Expression = unary(x, x => dn.pick_batch_elem(x, v))
  def pickBatchElems(x: Expression, v: UnsignedVector): Expression =
    unary(x, x => dn.pick_batch_elems(x, v.vector))

  def concatenateToBatch(v: ExpressionVector): Expression = vectory(v, dn.concatenate_to_batch)
  def concatenateToBatch(exprs: Expression*): Expression = concatenateToBatch(new ExpressionVector(exprs))

  def stridedSelect(x:Expression, strides:IntVector, from:IntVector, to:IntVector):Expression = {
    unary(x, x => dn.strided_select(x, strides.vector, from.vector, to.vector))
  }
  def stridedSelect(x:Expression, strides:Seq[Int], from:Seq[Int], to:Seq[Int]):Expression =
    stridedSelect(x, new IntVector(strides), new IntVector(from), new IntVector(to))

  def concatenateCols(v: ExpressionVector): Expression = vectory(v, dn.concatenate_cols)
  def concatenateCols(exprs: Expression*): Expression = concatenateCols(new ExpressionVector(exprs))

  def concatenate(v: ExpressionVector): Expression = vectory(v, dn.concatenate)
  def concatenate(exprs: Expression*): Expression = concatenate(new ExpressionVector(exprs))

  /* NOISE OPERATIONS */

  def noise(x: Expression, stddev: Float): Expression = unary(x, x => dn.noise(x, stddev))
  def dropout(x: Expression, p: Float): Expression = unary(x, x => dn.dropout(x, p))
  def dropoutDim(x: Expression, d: Long, p: Float): Expression = unary(x, x => dn.dropout_dim(x, d, p))
  def dropoutBatch(x: Expression, p: Float): Expression = unary(x, x => dn.dropout_batch(x, p))
  def blockDropout(x: Expression, p: Float): Expression = unary(x, x => dn.block_dropout(x, p))

  /* CONVOLUTION OPERATIONS */

  // These were commented out in the C++ code.
  //def conv1dNarrow(x: Expression, f: Expression): Expression = binary(x, f, dn.conv1d_narrow)
  //def conv1dWide(x: Expression, f: Expression): Expression = binary(x, f, dn.conv1d_wide)
  def filter1DNarrow(x: Expression, f: Expression): Expression = binary(x, f, dn.filter1d_narrow)
  def kMaxPooling(x: Expression, k: Long): Expression = unary(x, x => dn.kmax_pooling(x, k))
  def foldRows(x: Expression, nRows: Long = 2l): Expression = unary(x, x => dn.fold_rows(x, nRows))
  def sumDim(x: Expression, dims: UnsignedVector, b: Boolean = false): Expression = unary(x, x => dn.sum_dim(x, dims.vector, b))
  def sumCols(x: Expression): Expression = unary(x, dn.sum_cols)
  def sumRows(x: Expression): Expression = unary(x, dn.sum_rows)
  def averageCols(x: Expression): Expression = unary(x, dn.average_cols)
  def kmhNgram(x: Expression, n: Long): Expression = unary(x, x => dn.kmh_ngram(x, n))

  // In the C++ code, is_valid has a default value of true. Scala won't let you have two overloaded
  // methods with default values, so I just got rid of the default value here.
  // TODO(joelgrus): write tests for these
  def conv2d(x: Expression, f: Expression, stride: UnsignedVector, isValid: Boolean) =
    new Expression(dn.conv2d(x.expr, f.expr, stride.vector, isValid), Seq(x, f, stride))
  def conv2d(x: Expression, f: Expression, stride: UnsignedVector) =
    new Expression(dn.conv2d(x.expr, f.expr, stride.vector, true), Seq(x, f, stride))

  def conv2d(x: Expression, f: Expression, b: Expression, stride: UnsignedVector, isValid: Boolean) =
    new Expression(dn.conv2d(x.expr, f.expr, b.expr, stride.vector, isValid), Seq(x, f, b, stride))
  def conv2d(x: Expression, f: Expression, b: Expression, stride: UnsignedVector) =
    new Expression(dn.conv2d(x.expr, f.expr, b.expr, stride.vector, true), Seq(x, f, b, stride))

  /* TENSOR OPERATIONS */

  def contract3d1d(x: Expression, y: Expression): Expression = binary(x, y, dn.contract3d_1d)
  def contract3d1d1d(x: Expression, y: Expression, z: Expression): Expression = {
    Seq(x, y, z).foreach(_.ensureFresh)
    new Expression(dn.contract3d_1d_1d(x.expr, y.expr, z.expr), Seq(x, y, z))
  }
  def contract3d1d1d(x: Expression, y: Expression, z: Expression, b: Expression): Expression = {
    Seq(x, y, z, b).foreach(_.ensureFresh)
    new Expression(dn.contract3d_1d_1d(x.expr, y.expr, z.expr, b.expr), Seq(x, y, z, b))
  }
  def contract3d1d(x: Expression, y: Expression, b: Expression): Expression = {
    Seq(x, y, b).foreach(_.ensureFresh)
    new Expression(dn.contract3d_1d(x.expr, y.expr, b.expr), Seq(x, y, b))
  }

  /* LINEAR ALGEBRA OPERATIONS */

  def inverse(x: Expression): Expression = unary(x, dn.inverse)
  def logdet(x: Expression): Expression = unary(x, dn.logdet)
  def traceOfProduct(x: Expression, y: Expression): Expression = binary(x, y, dn.trace_of_product)

  /* NORMALIZATION OPERATIONS */

  def layerNorm(x: Expression, g: Expression, b: Expression): Expression = {
    Seq(x, g, b).foreach(_.ensureFresh)
    new Expression(dn.layer_norm(x.expr, g.expr, b.expr), Seq(x, g, b))
  }
  def weightNorm(w: Expression, g: Expression): Expression = binary(w, g, dn.weight_norm)

  /** Augment numbers so that they can do arithmetic with expressions. */
  implicit class ImplicitNumerics[T](x: T)(implicit n: Numeric[T]) {
    import n._
    def +(e: Expression): Expression = Expression.exprPlus(x.toFloat, e)
    def *(e: Expression): Expression = Expression.exprTimes(x.toFloat, e)
    def -(e: Expression): Expression = Expression.exprMinus(x.toFloat, e)
  }
}
