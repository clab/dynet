#include "dynet/expr.h"

#include <initializer_list>

#include "dynet/nodes.h"
#include "dynet/nodes-conv.h"

namespace dynet { namespace expr {

using std::vector;

Expression input(ComputationGraph& g, real s) { return Expression(&g, g.add_input(s)); }
Expression input(ComputationGraph& g, const real *ps) { return Expression(&g, g.add_input(ps)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>& data) { return Expression(&g, g.add_input(d, data)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>* pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<unsigned int>& ids, const vector<float>& data, float defdata) { return Expression(&g, g.add_input(d, ids, data, defdata)); }
Expression const_parameter(ComputationGraph& g, Parameter p) { return Expression(&g, g.add_const_parameters(p)); }
Expression parameter(ComputationGraph& g, Parameter p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }
Expression lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices) { return Expression(&g, g.add_lookup(p, indices)); }
Expression lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices) { return Expression(&g, g.add_lookup(p, pindices)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index) { return Expression(&g, g.add_const_lookup(p, index)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex) { return Expression(&g, g.add_const_lookup(p, pindex)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices) { return Expression(&g, g.add_const_lookup(p, indices)); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices) { return Expression(&g, g.add_const_lookup(p, pindices)); }
Expression zeroes(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<Zeroes>(d)); }
Expression random_normal(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<RandomNormal>(d)); }
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale) { return Expression(&g, g.add_function<RandomBernoulli>({}, d, p, scale)); }
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right) { return Expression(&g, g.add_function<RandomUniform>({}, d, left, right)); }

// identity function, but derivative is not propagated through it
Expression nobackprop(const Expression& x) { return Expression(x.pg, x.pg->add_function<NoBackprop>({x.i})); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>({x.i})); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Sum>({x.i, y.i})); }
Expression operator+(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantPlusX>({y.i}, x)); }
Expression operator+(const Expression& x, real y) { return y+x; }
Expression operator-(const Expression& x, const Expression& y) { return x+(-y); }
Expression operator-(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantMinusX>({y.i}, x)); }
Expression operator-(const Expression& x, real y) { return -(y-x); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>({x.i, y.i})); }
Expression operator*(const Expression& x, float y) { return Expression(x.pg, x.pg->add_function<ConstScalarMultiply>({x.i}, y)); }
Expression cmult(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<CwiseMultiply>({x.i, y.i}));}
Expression cdiv(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseQuotient>({x.i, y.i})); }
Expression colwise_add(const Expression& x, const Expression& bias) { return Expression(x.pg, x.pg->add_function<AddVectorToAllColumns>({x.i, bias.i})); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>({x.i, y.i, z.i})); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>({x.i, y.i, z.i, b.i})); }
Expression contract3d_1d(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i})); }
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i, b.i})); }

Expression sqrt(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sqrt>({x.i})); }
Expression erf(const Expression& x) { return Expression(x.pg, x.pg->add_function<Erf>({x.i})); }
Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>({x.i})); }
Expression lgamma(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogGamma>({x.i})); }
Expression log(const Expression& x) { return Expression(x.pg, x.pg->add_function<Log>({x.i})); }
Expression exp(const Expression& x) { return Expression(x.pg, x.pg->add_function<Exp>({x.i})); }
Expression square(const Expression& x) { return Expression(x.pg, x.pg->add_function<Square>({x.i})); }
Expression cube(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cube>({x.i})); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>({x.i})); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>({x.i})); }
Expression hinge(const Expression& x, unsigned index, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, index, m)); }
Expression hinge(const Expression& x, const unsigned* pindex, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindex, m)); }
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, indices, m)); }
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindices, m)); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>({x.i})); }
Expression log_softmax(const Expression& x, const vector<unsigned>& d) { return Expression(x.pg, x.pg->add_function<RestrictedLogSoftmax>({x.i}, d)); }
Expression sparsemax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sparsemax>({x.i})); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>& target_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>({x.i}, target_support)); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>* ptarget_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>({x.i}, ptarget_support)); }
Expression softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Softmax>({x.i})); }
Expression softsign(const Expression& x) { return Expression(x.pg, x.pg->add_function<SoftSign>({x.i})); }
Expression pow(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Pow>({x.i, y.i})); }
Expression min(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Min>({x.i, y.i})); }
Expression max(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Max>({x.i, y.i})); }
Expression noise(const Expression& x, real stddev) { return Expression(x.pg, x.pg->add_function<GaussianNoise>({x.i}, stddev)); }
Expression dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<Dropout>({x.i}, p)); }
Expression block_dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<BlockDropout>({x.i}, p)); }

Expression reshape(const Expression& x, const Dim& d) { return Expression(x.pg, x.pg->add_function<Reshape>({x.i}, d)); }
Expression transpose(const Expression& x) { return Expression(x.pg, x.pg->add_function<Transpose>({x.i})); }
Expression select_rows(const Expression& x, const vector<unsigned>& rows) { return Expression(x.pg, x.pg->add_function<SelectRows>({x.i}, rows)); }
Expression select_rows(const Expression& x, const vector<unsigned>* prows) { return Expression(x.pg, x.pg->add_function<SelectRows>({x.i}, prows)); }
Expression select_cols(const Expression& x, const vector<unsigned>& cols) { return Expression(x.pg, x.pg->add_function<SelectCols>({x.i}, cols)); }
Expression select_cols(const Expression& x, const vector<unsigned>* pcols) { return Expression(x.pg, x.pg->add_function<SelectCols>({x.i}, pcols)); }
Expression inverse(const Expression& x) { return Expression(x.pg, x.pg->add_function<MatrixInverse>({x.i})); }
Expression logdet(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogDet>({x.i})); }

Expression trace_of_product(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<TraceOfProduct>({x.i, y.i}));}

Expression squared_norm(const Expression& x) { return Expression(x.pg, x.pg->add_function<SquaredNorm>({x.i})); }

Expression dot_product(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<DotProduct>({x.i, y.i})); }
Expression squared_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>({x.i, y.i})); }
Expression huber_distance(const Expression& x, const Expression& y, real c) { return Expression(x.pg, x.pg->add_function<HuberDistance>({x.i, y.i}, c)); }
Expression l1_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<L1Distance>({x.i, y.i})); }
Expression binary_log_loss(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>({x.i,y.i})); }
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m) { return Expression(x.pg, x.pg->add_function<PairwiseRankLoss>({x.i, y.i}, m)); }
Expression poisson_loss(const Expression& x, unsigned y) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, y)); }
Expression poisson_loss(const Expression& x, const unsigned* py) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, py)); }

Expression conv1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DNarrow>({x.i, f.i})); }
Expression conv1d_wide(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DWide>({x.i, f.i})); }
Expression filter1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Filter1DNarrow>({x.i, f.i})); }
Expression kmax_pooling(const Expression& x, unsigned k) { return Expression(x.pg, x.pg->add_function<KMaxPooling>({x.i}, k)); }
Expression fold_rows(const Expression& x, unsigned nrows) { return Expression(x.pg, x.pg->add_function<FoldRows>({x.i}, nrows)); }

Expression pick(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v)); }
Expression pick(const Expression& x, const vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v)); }
Expression pick(const Expression& x, unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv)); }
Expression pick(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv)); }

Expression pickrange(const Expression& x, unsigned v, unsigned u) { return Expression(x.pg, x.pg->add_function<PickRange>({x.i}, v, u)); }

Expression pickneglogsoftmax(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v)); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v)); }
Expression pickneglogsoftmax(const Expression& x, unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, pv)); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, pv)); }

Expression average_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<AverageColumns>({x.i})); }
Expression sum_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumColumns>({x.i})); }

Expression sum_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumBatches>({x.i})); }

Expression kmh_ngram(const Expression& x, unsigned n) { return Expression(x.pg, x.pg->add_function<KMHNGram>({x.i}, n)); }


} }
