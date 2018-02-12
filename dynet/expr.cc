#include "dynet/expr.h"

#include <initializer_list>

#include "dynet/nodes.h"
#include "dynet/devices.h"

namespace dynet {

using std::vector;

std::string Expression::get_device_name() const {
  if (pg->nodes[i]->device == nullptr)
    throw std::runtime_error("Unknown device for node:" + std::to_string(i));
  return pg->nodes[i]->device->name;
}

Expression input(ComputationGraph& g, real s, Device *device) { return Expression(&g, g.add_input(s, device)); }
Expression input(ComputationGraph& g, const real *ps, Device *device) { return Expression(&g, g.add_input(ps, device)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>& data, Device *device) { return Expression(&g, g.add_input(d, data, device)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>* pdata, Device *device) { return Expression(&g, g.add_input(d, pdata, device)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<unsigned int>& ids, const vector<float>& data, float defdata, Device *device) { return Expression(&g, g.add_input(d, ids, data, device, defdata)); }
Expression const_parameter(ComputationGraph& g, Parameter p) { return Expression(&g, g.add_const_parameters(p)); }
Expression const_parameter(ComputationGraph& g, LookupParameter p) { return Expression(&g, g.add_const_parameters(p)); }
Expression parameter(ComputationGraph& g, Parameter p) { return Expression(&g, g.add_parameters(p)); }
Expression parameter(ComputationGraph& g, LookupParameter p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }
Expression lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices) { return Expression(&g, g.add_lookup(p, indices)); }
Expression lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices) { return Expression(&g, g.add_lookup(p, pindices)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index) { return Expression(&g, g.add_const_lookup(p, index)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex) { return Expression(&g, g.add_const_lookup(p, pindex)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices) { return Expression(&g, g.add_const_lookup(p, indices)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices) { return Expression(&g, g.add_const_lookup(p, pindices)); }
Expression zeros(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<Constant>(d, 0.f)); }
// Expression zeroes(ComputationGraph& g, const Dim& d) {return zeros(g, d);}
Expression ones(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<Constant>(d, 1.f)); }
Expression constant(ComputationGraph& g, const Dim& d, float val) { return Expression(&g, g.add_function<Constant>(d, val)); }
Expression random_normal(ComputationGraph& g, const Dim& d, float mean, float stddev) { return Expression(&g, g.add_function<RandomNormal>(d, mean, stddev)); }
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale) { return Expression(&g, g.add_function<RandomBernoulli>({}, d, p, scale)); }
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right) { return Expression(&g, g.add_function<RandomUniform>({}, d, left, right)); }
Expression random_gumbel(ComputationGraph& g, const Dim& d, real mu, real beta) { return Expression(&g, g.add_function<RandomGumbel>({}, d, mu, beta)); }

Expression nobackprop(const Expression& x) { return Expression(x.pg, x.pg->add_function<NoBackprop>({x.i})); }
Expression flip_gradient(const Expression& x) { return Expression(x.pg, x.pg->add_function<ScaleGradient>({x.i}, -1.f)); }
Expression scale_gradient(const Expression& x, float lambd) { return Expression(x.pg, x.pg->add_function<ScaleGradient>({x.i}, lambd)); }

Expression argmax(const Expression& x, ArgmaxGradient gradient_mode) { return Expression(x.pg, x.pg->add_function<Argmax>({x.i}, 0, (gradient_mode==straight_through_gradient))); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>({x.i})); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseSum>({x.i, y.i}));}

Expression operator+(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantPlusX>({y.i}, x)); }
Expression operator+(const Expression& x, real y) { return y + x; }
Expression operator-(const Expression& x, const Expression& y) { return x + (-y); }
Expression operator-(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantMinusX>({y.i}, x)); }
Expression operator-(const Expression& x, real y) { return -(y - x); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>({x.i, y.i})); }
Expression operator*(const Expression& x, float y) { return Expression(x.pg, x.pg->add_function<ConstScalarMultiply>({x.i}, y)); }
Expression cmult(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseMultiply>({x.i, y.i})); }
Expression cdiv(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseQuotient>({x.i, y.i})); }
Expression colwise_add(const Expression& x, const Expression& bias) { return Expression(x.pg, x.pg->add_function<AddVectorToAllColumns>({x.i, bias.i})); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>({x.i, y.i, z.i})); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>({x.i, y.i, z.i, b.i})); }
Expression contract3d_1d(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i})); }
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i, b.i})); }

Expression sqrt(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sqrt>({x.i})); }
Expression abs(const Expression& x) { return Expression(x.pg, x.pg->add_function<Abs>({x.i})); }
Expression erf(const Expression& x) { return Expression(x.pg, x.pg->add_function<Erf>({x.i})); }
Expression sin(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sin>({x.i})); }
Expression cos(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cos>({x.i})); }
Expression tan(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tan>({x.i})); }
Expression asin(const Expression& x) { return Expression(x.pg, x.pg->add_function<Asin>({x.i})); }
Expression acos(const Expression& x) { return Expression(x.pg, x.pg->add_function<Acos>({x.i})); }
Expression atan(const Expression& x) { return Expression(x.pg, x.pg->add_function<Atan>({x.i})); }
Expression sinh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sinh>({x.i})); }
Expression cosh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cosh>({x.i})); }
Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>({x.i})); }
Expression asinh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Asinh>({x.i})); }
Expression acosh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Acosh>({x.i})); }
Expression atanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Atanh>({x.i})); }
Expression log_sigmoid(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSigmoid>({x.i})); }
Expression lgamma(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogGamma>({x.i})); }
Expression log(const Expression& x) { return Expression(x.pg, x.pg->add_function<Log>({x.i})); }
Expression exp(const Expression& x) { return Expression(x.pg, x.pg->add_function<Exp>({x.i})); }
Expression square(const Expression& x) { return Expression(x.pg, x.pg->add_function<Square>({x.i})); }
Expression cube(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cube>({x.i})); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>({x.i})); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>({x.i})); }
Expression elu(const Expression& x, float alpha) { return Expression(x.pg, x.pg->add_function<ExponentialLinearUnit>({x.i}, 1.0, alpha)); }
Expression selu(const Expression& x) { return Expression(x.pg, x.pg->add_function<ExponentialLinearUnit>({x.i}, 1.0507009873554804934193349852946, 1.6732632423543772848170429916717)); }
Expression silu(const Expression& x, float beta) { return Expression(x.pg, x.pg->add_function<SigmoidLinearUnit>({x.i}, beta)); }
Expression hinge(const Expression& x, unsigned index, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, index, m)); }
Expression hinge(const Expression& x, const unsigned* pindex, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindex, m)); }
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, indices, m)); }
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindices, m)); }
Expression hinge_dim(const Expression& x, const std::vector<unsigned> & index, unsigned d, float m) { return Expression(x.pg, x.pg->add_function<HingeDim>({x.i}, index, d, m)); }
Expression hinge_dim(const Expression& x, const std::vector<unsigned> * pindex, unsigned d, float m) { return Expression(x.pg, x.pg->add_function<HingeDim>({x.i}, pindex, d, m)); }
Expression hinge_dim(const Expression& x, const std::vector<std::vector<unsigned> > & indices, unsigned d, float m) { return Expression(x.pg, x.pg->add_function<HingeDim>({x.i}, indices, d, m)); }
Expression hinge_dim(const Expression& x, const std::vector<std::vector<unsigned> > * pindices, unsigned d, float m) { return Expression(x.pg, x.pg->add_function<HingeDim>({x.i}, pindices, d, m)); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>({x.i})); }
Expression log_softmax(const Expression& x, const vector<unsigned>& d) { return Expression(x.pg, x.pg->add_function<RestrictedLogSoftmax>({x.i}, d)); }
Expression logsumexp_dim(const Expression& x, unsigned d) { return Expression(x.pg, x.pg->add_function<LogSumExpDimension>({x.i}, d)); }
Expression sparsemax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sparsemax>({x.i})); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>& target_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>({x.i}, target_support)); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>* ptarget_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>({x.i}, ptarget_support)); }
Expression softmax(const Expression& x, unsigned d) { return Expression(x.pg, x.pg->add_function<Softmax>({x.i}, d)); }
Expression constrained_softmax(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<ConstrainedSoftmax>({x.i, y.i})); }
Expression softsign(const Expression& x) { return Expression(x.pg, x.pg->add_function<SoftSign>({x.i})); }
Expression pow(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Pow>({x.i, y.i})); }
Expression min(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Min>({x.i, y.i})); }
Expression max(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Max>({x.i, y.i})); }
Expression noise(const Expression& x, real stddev) { return Expression(x.pg, x.pg->add_function<GaussianNoise>({x.i}, stddev)); }
Expression dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<Dropout>({x.i}, p)); }
Expression dropout_batch(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<DropoutBatch>({x.i}, p)); }
Expression dropout_dim(const Expression& x, unsigned d, real p) { return Expression(x.pg, x.pg->add_function<DropoutDim>({x.i}, d, p)); }
Expression block_dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<BlockDropout>({x.i}, p)); }

Expression reshape(const Expression& x, const Dim& d) { return Expression(x.pg, x.pg->add_function<Reshape>({x.i}, d)); }
Expression transpose(const Expression& x, const vector<unsigned>& dims) { return Expression(x.pg, x.pg->add_function<Transpose>({x.i}, dims)); }
Expression select_rows(const Expression& x, const vector<unsigned>& rows) { return Expression(x.pg, x.pg->add_function<SelectRows>({x.i}, rows)); }
Expression select_rows(const Expression& x, const vector<unsigned>* prows) { return Expression(x.pg, x.pg->add_function<SelectRows>({x.i}, prows)); }
Expression select_cols(const Expression& x, const vector<unsigned>& cols) { return Expression(x.pg, x.pg->add_function<SelectCols>({x.i}, cols)); }
Expression select_cols(const Expression& x, const vector<unsigned>* pcols) { return Expression(x.pg, x.pg->add_function<SelectCols>({x.i}, pcols)); }
Expression inverse(const Expression& x) { return Expression(x.pg, x.pg->add_function<MatrixInverse>({x.i})); }
Expression logdet(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogDet>({x.i})); }

Expression trace_of_product(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<TraceOfProduct>({x.i, y.i}));}

Expression squared_norm(const Expression& x) { return Expression(x.pg, x.pg->add_function<SquaredNorm>({x.i})); }

Expression l2_norm(const Expression& x) { return Expression(x.pg, x.pg->add_function<L2Norm>({x.i})); }

Expression dot_product(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<DotProduct>({x.i, y.i})); }
Expression circ_conv(const Expression& u, const Expression& v) { return Expression(u.pg, u.pg->add_function<CircularConvolution>({u.i, v.i})); }
Expression circ_corr(const Expression& u, const Expression& v) { return Expression(u.pg, u.pg->add_function<CircularCorrelation>({u.i, v.i})); }
Expression squared_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>({x.i, y.i})); }
Expression huber_distance(const Expression& x, const Expression& y, real c) { return Expression(x.pg, x.pg->add_function<HuberDistance>({x.i, y.i}, c)); }
Expression l1_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<L1Distance>({x.i, y.i})); }
Expression binary_log_loss(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>({x.i, y.i})); }
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m) { return Expression(x.pg, x.pg->add_function<PairwiseRankLoss>({x.i, y.i}, m)); }
Expression poisson_loss(const Expression& x, unsigned y) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, y)); }
Expression poisson_loss(const Expression& x, const unsigned* py) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, py)); }

//Expression conv1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DNarrow>({x.i, f.i})); }
//Expression conv1d_wide(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DWide>({x.i, f.i})); }
Expression filter1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Filter1DNarrow>({x.i, f.i})); }
Expression kmax_pooling(const Expression& x, unsigned k, unsigned d) { return Expression(x.pg, x.pg->add_function<KMaxPooling>({x.i}, k, d)); }
Expression fold_rows(const Expression& x, unsigned nrows) { return Expression(x.pg, x.pg->add_function<FoldRows>({x.i}, nrows)); }
Expression conv2d(const Expression& x, const Expression& f, const std::vector<unsigned>& stride, bool is_valid) { return Expression(x.pg, x.pg->add_function<Conv2D>({x.i, f.i}, stride, is_valid)); }
Expression conv2d(const Expression& x, const Expression& f, const Expression& b, const std::vector<unsigned>& stride, bool is_valid) {
  return Expression(x.pg, x.pg->add_function<Conv2D>({x.i, f.i, b.i}, stride, is_valid));
}
Expression maxpooling2d(const Expression& x, const std::vector<unsigned>& ksize, const std::vector<unsigned>& stride, bool is_valid) {
  return Expression(x.pg, x.pg->add_function<MaxPooling2D>({x.i}, ksize, stride, is_valid));
}


Expression pick(const Expression& x, unsigned v, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v, d)); }
Expression pick(const Expression& x, const vector<unsigned> & v, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v, d)); }
Expression pick(const Expression& x, const unsigned* pv, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv, d)); }
Expression pick(const Expression& x, const vector<unsigned> * pv, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv, d)); }

Expression pick_batch_elem(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickBatchElements>({x.i}, v)); }
Expression pick_batch_elems(const Expression& x, const std::vector<unsigned>& v) { return Expression(x.pg, x.pg->add_function<PickBatchElements>({x.i}, v)); }
Expression pick_batch_elem(const Expression& x, const unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickBatchElements>({x.i}, pv)); }
Expression pick_batch_elems(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickBatchElements>({x.i}, pv)); }

Expression pick_range(const Expression& x, unsigned v, unsigned u, unsigned d) { return Expression(x.pg, x.pg->add_function<PickRange>({x.i}, v, u, d)); }
Expression pickrange(const Expression& x, unsigned v, unsigned u) {
  std::cerr << "WARNING: The function naming pickrange() has been deprecated. Please use pick_range() instead." << std::endl;
  return Expression(x.pg, x.pg->add_function<PickRange>({x.i}, v, u, 0));
}

Expression strided_select(const Expression& x, const std::vector<int>& strides, const std::vector<int>& range_from, const std::vector<int>& range_to) { return Expression(x.pg, x.pg->add_function<StridedSelect>({x.i}, strides, range_from, range_to)); }

Expression pickneglogsoftmax(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v)); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v)); }
Expression pickneglogsoftmax(const Expression& x, const unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, pv)); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, pv)); }

Expression average_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<AverageColumns>({x.i})); }
Expression sum_dim(const Expression& x, const vector<unsigned>& dims, bool b) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, dims, b)); }
Expression sum_rows(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, vector<unsigned>({0}), false)); }
Expression sum_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, vector<unsigned>({1}), false)); }
Expression sum_elems(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumElements>({x.i})); }

Expression cumsum(const Expression& x, unsigned d) { return Expression(x.pg, x.pg->add_function<CumulativeSum>({x.i}, d)); }

Expression sum_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, vector<unsigned>(), true)); }

Expression moment_dim(const Expression& x, const vector<unsigned>& dims, unsigned r, bool b, unsigned n) { return Expression(x.pg, x.pg->add_function<MomentDimension>({x.i}, dims, r, b, n)); }
Expression mean_dim(const Expression& x, const vector<unsigned>& dims, bool b, unsigned n) { return Expression(x.pg, x.pg->add_function<MomentDimension>({x.i}, dims, 1, b, n)); }
Expression std_dim(const Expression& x, const vector<unsigned>& dims, bool b, unsigned n) { return Expression(x.pg, x.pg->add_function<StdDimension>({x.i}, dims, b, n)); }

Expression moment_elems(const Expression& x, unsigned r) { 
  vector<unsigned> dims(x.dim().nd);
  for(size_t i = 0; i < dims.size(); ++i) dims[i] = i;
  return Expression(x.pg, x.pg->add_function<MomentDimension>({x.i}, dims, r, false));
}
Expression mean_elems(const Expression& x) { 
  vector<unsigned> dims(x.dim().nd);
  for(size_t i = 0; i < dims.size(); ++i) dims[i] = i;
  return Expression(x.pg, x.pg->add_function<MomentDimension>({x.i}, dims, 1, false));
}
Expression std_elems(const Expression& x) { 
  vector<unsigned> dims(x.dim().nd);
  for(size_t i = 0; i < dims.size(); ++i) dims[i] = i;
  return Expression(x.pg, x.pg->add_function<StdDimension>({x.i}, dims, false));
}
Expression mean_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<MomentDimension>({x.i}, vector<unsigned>(), 1, true)); }
Expression moment_batches(const Expression& x, unsigned r) { return Expression(x.pg, x.pg->add_function<MomentDimension>({x.i}, vector<unsigned>(), r, true)); }
Expression std_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<StdDimension>({x.i}, vector<unsigned>(), true)); }

Expression kmh_ngram(const Expression& x, unsigned n) { return Expression(x.pg, x.pg->add_function<KMHNGram>({x.i}, n)); }

Expression max_dim(const Expression& x, unsigned d) { return Expression(x.pg, x.pg->add_function<MaxDimension>({x.i}, d)); }
Expression min_dim(const Expression& x, unsigned d) { return Expression(x.pg, x.pg->add_function<MinDimension>({x.i}, d)); }

Expression layer_norm(const Expression& x, const Expression& g, const Expression& b){
    Expression mu = mean_elems(x);
    Expression x_centered= x - mu;
    Expression sigma = std_elems(x);
    return cmult(g, cdiv(x_centered,sigma + 1e-8)) + b;
}

Expression weight_norm(const Expression& w, const Expression& g){return Expression(w.pg, w.pg->add_function<WeightNormalization>({w.i,g.i}));}

Expression vanilla_lstm_gates_concat(const std::vector<Expression>& x_t, const Expression& h_tm1, const Expression& Wx, const Expression& Wh, const Expression& b, real weightnoise_std){
  std::vector<VariableIndex> xis(x_t.size() + 4);
  unsigned i=0;
  std::vector<Expression>::const_iterator it;
  for ( it=x_t.begin(); it!=x_t.end(); ++it){
    xis[i] = it->i;
    i++;
  }
  xis[x_t.size()] = h_tm1.i;
  xis[x_t.size()+1] = Wx.i;
  xis[x_t.size()+2] = Wh.i;
  xis[x_t.size()+3] = b.i;
  return Expression(h_tm1.pg, h_tm1.pg->add_function<VanillaLSTMGates>(xis, false, weightnoise_std));
}
Expression vanilla_lstm_gates(const Expression& x_t, const Expression& h_tm1, const Expression& Wx, const Expression& Wh, const Expression& b, real weightnoise_std){
  return vanilla_lstm_gates_concat({x_t}, h_tm1, Wx, Wh, b, weightnoise_std);
}
Expression vanilla_lstm_gates_dropout_concat(const std::vector<Expression>& x_t, const Expression& h_tm1, const Expression& Wx, const Expression& Wh, const Expression& b, const Expression& dropout_mask_x, const Expression& dropout_mask_h, real weightnoise_std){
  std::vector<VariableIndex> xis(x_t.size() + 6);
  unsigned i=0;
  std::vector<Expression>::const_iterator it;
  for ( it=x_t.begin(); it!=x_t.end(); ++it){
    xis[i] = it->i;
    i++;
  }
  xis[x_t.size()] = h_tm1.i;
  xis[x_t.size()+1] = Wx.i;
  xis[x_t.size()+2] = Wh.i;
  xis[x_t.size()+3] = b.i;
  xis[x_t.size()+4] = dropout_mask_x.i;
  xis[x_t.size()+5] = dropout_mask_h.i;
  return Expression(h_tm1.pg, h_tm1.pg->add_function<VanillaLSTMGates>(xis, true, weightnoise_std));
}
Expression vanilla_lstm_gates_dropout(const Expression& x_t, const Expression& h_tm1, const Expression& Wx, const Expression& Wh, const Expression& b, const Expression& dropout_mask_x, const Expression& dropout_mask_, real weightnoise_std){
  return vanilla_lstm_gates_dropout_concat({x_t}, h_tm1, Wx, Wh, b, dropout_mask_x, dropout_mask_, weightnoise_std);
}
Expression vanilla_lstm_c(const Expression& c_tm1, const Expression& gates_t){
  return Expression(c_tm1.pg, c_tm1.pg->add_function<VanillaLSTMC>({c_tm1.i, gates_t.i}));
}
Expression vanilla_lstm_h(const Expression& c_t, const Expression& gates_t){
  return Expression(c_t.pg, c_t.pg->add_function<VanillaLSTMH>({c_t.i, gates_t.i}));
}

Expression to_device(const Expression & x, Device *device) {
  DYNET_ASSERT(x.pg->nodes[x.i]->device != device, "It is unnecessary to perform to_device operation in the same devices");
  return Expression(x.pg, x.pg->add_function<ToDevice>({x.i}, device));
}

}  // namespace dynet
