#include "dynet/expr.h"

#include <initializer_list>

#include "dynet/nodes.h"
#include "dynet/nodes-conv.h"

namespace dynet {
namespace expr {

using std::vector;

Expression input(ComputationGraph& g, real s) { return Expression(&g, g.add_input(s), "input(ComputationGraph& g, real s)"); }
Expression input(ComputationGraph& g, const real *ps) { return Expression(&g, g.add_input(ps), "input(ComputationGraph& g, const real *ps)"); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>& data) { return Expression(&g, g.add_input(d, data), "input(ComputationGraph& g, const Dim& d, const vector<float>& data)"); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>* pdata) { return Expression(&g, g.add_input(d, pdata), "input(ComputationGraph& g, const Dim& d, const vector<float>* pdata)"); }
Expression input(ComputationGraph& g, const Dim& d, const vector<unsigned int>& ids, const vector<float>& data, float defdata) { return Expression(&g, g.add_input(d, ids, data, defdata), "input(ComputationGraph& g, const Dim& d, const vector<unsigned int>& ids, const vector<float>& data, float defdata)"); }
Expression const_parameter(ComputationGraph& g, Parameter p) { return Expression(&g, g.add_const_parameters(p), "const_parameter(ComputationGraph& g, Parameter p)"); }
Expression parameter(ComputationGraph& g, Parameter p) { return Expression(&g, g.add_parameters(p), "parameter(ComputationGraph& g, Parameter p)"); }
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index) { return Expression(&g, g.add_lookup(p, index), "lookup(ComputationGraph& g, LookupParameter p, unsigned index)"); }
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex), "lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex)"); }
Expression lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices) { return Expression(&g, g.add_lookup(p, indices), "lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices)"); }
Expression lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices) { return Expression(&g, g.add_lookup(p, pindices), "lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices)"); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index) { return Expression(&g, g.add_const_lookup(p, index), "const_lookup(ComputationGraph& g, LookupParameter p, unsigned index)"); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex) { return Expression(&g, g.add_const_lookup(p, pindex), "const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex)"); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices) { return Expression(&g, g.add_const_lookup(p, indices), "const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>& indices)"); }
Expression const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices) { return Expression(&g, g.add_const_lookup(p, pindices), "const_lookup(ComputationGraph& g, LookupParameter p, const vector<unsigned>* pindices)"); }
Expression zeroes(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<Zeroes>(d), "zeroes(ComputationGraph& g, const Dim& d)"); }
Expression random_normal(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<RandomNormal>(d), "random_normal(ComputationGraph& g, const Dim& d)"); }
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale) { return Expression(&g, g.add_function<RandomBernoulli>({}, d, p, scale), "random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale)"); }
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right) { return Expression(&g, g.add_function<RandomUniform>({}, d, left, right), "random_uniform(ComputationGraph& g, const Dim& d, real left, real right)"); }

// identity function, but derivative is not propagated through it
Expression nobackprop(const Expression& x) { return Expression(x.pg, x.pg->add_function<NoBackprop>({x.i}), "nobackprop(const Expression& x)"); }
// identity function, but derivative is propagated as negative
Expression flip_gradient(const Expression& x) { return Expression(x.pg, x.pg->add_function<FlipGradient>({x.i}), "flip_gradient(const Expression& x)"); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>({x.i}), "operator-(const Expression& x)"); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Sum>({x.i, y.i}), "operator+(const Expression& x, const Expression& y)"); }
Expression operator+(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantPlusX>({y.i}, x), "operator+(real x, const Expression& y)"); }
Expression operator+(const Expression& x, real y) { return y + x; }
Expression operator-(const Expression& x, const Expression& y) { return x + (-y); }
Expression operator-(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantMinusX>({y.i}, x), "operator-(real x, const Expression& y)"); }
Expression operator-(const Expression& x, real y) { return -(y - x); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>({x.i, y.i}), "operator*(const Expression& x, const Expression& y)"); }
Expression operator*(const Expression& x, float y) { return Expression(x.pg, x.pg->add_function<ConstScalarMultiply>({x.i}, y), "operator*(const Expression& x, float y)"); }
Expression cmult(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<CwiseMultiply>({x.i, y.i}), "cmult(const Expression& x, const Expression& y)");}
Expression cdiv(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseQuotient>({x.i, y.i}), "cdiv(const Expression& x, const Expression& y)"); }
Expression colwise_add(const Expression& x, const Expression& bias) { return Expression(x.pg, x.pg->add_function<AddVectorToAllColumns>({x.i, bias.i}), "colwise_add(const Expression& x, const Expression& bias)"); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>({x.i, y.i, z.i}), "contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z)"); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>({x.i, y.i, z.i, b.i}), "contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b)"); }
Expression contract3d_1d(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i}), "contract3d_1d(const Expression& x, const Expression& y)"); }
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i, b.i}), "contract3d_1d(const Expression& x, const Expression& y, const Expression& b)"); }

Expression sqrt(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sqrt>({x.i}), "sqrt(const Expression& x)"); }
Expression erf(const Expression& x) { return Expression(x.pg, x.pg->add_function<Erf>({x.i}), "erf(const Expression& x)"); }
Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>({x.i}), "tanh(const Expression& x)"); }
Expression lgamma(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogGamma>({x.i}), "lgamma(const Expression& x)"); }
Expression log(const Expression& x) { return Expression(x.pg, x.pg->add_function<Log>({x.i}), "log(const Expression& x)"); }
Expression exp(const Expression& x) { return Expression(x.pg, x.pg->add_function<Exp>({x.i}), "exp(const Expression& x)"); }
Expression square(const Expression& x) { return Expression(x.pg, x.pg->add_function<Square>({x.i}), "square(const Expression& x)"); }
Expression cube(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cube>({x.i}), "cube(const Expression& x)"); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>({x.i}), "logistic(const Expression& x)"); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>({x.i}), "rectify(const Expression& x)"); }
Expression hinge(const Expression& x, unsigned index, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, index, m), "hinge(const Expression& x, unsigned index, float m)"); }
Expression hinge(const Expression& x, const unsigned* pindex, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindex, m), "hinge(const Expression& x, const unsigned* pindex, float m)"); }
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, indices, m), "hinge(const Expression& x, const std::vector<unsigned> & indices, float m)"); }
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindices, m), "hinge(const Expression& x, const std::vector<unsigned> * pindices, float m)"); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>({x.i}), "log_softmax(const Expression& x)"); }
Expression log_softmax(const Expression& x, const vector<unsigned>& d) { return Expression(x.pg, x.pg->add_function<RestrictedLogSoftmax>({x.i}, d), "log_softmax(const Expression& x, const vector<unsigned>& d)"); }
Expression sparsemax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sparsemax>({x.i}), "sparsemax(const Expression& x)"); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>& target_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>({x.i}, target_support), "sparsemax_loss(const Expression& x, const vector<unsigned>& target_support)"); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>* ptarget_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>({x.i}, ptarget_support), "sparsemax_loss(const Expression& x, const vector<unsigned>* ptarget_support)"); }
Expression softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Softmax>({x.i}), "softmax(const Expression& x)"); }
Expression softsign(const Expression& x) { return Expression(x.pg, x.pg->add_function<SoftSign>({x.i}), "softsign(const Expression& x)"); }
Expression pow(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Pow>({x.i, y.i}), "pow(const Expression& x, const Expression& y)"); }
Expression min(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Min>({x.i, y.i}), "min(const Expression& x, const Expression& y)"); }
Expression max(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Max>({x.i, y.i}), "max(const Expression& x, const Expression& y)"); }
Expression noise(const Expression& x, real stddev) { return Expression(x.pg, x.pg->add_function<GaussianNoise>({x.i}, stddev), "noise(const Expression& x, real stddev)"); }
Expression dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<Dropout>({x.i}, p), "dropout(const Expression& x, real p)"); }
Expression block_dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<BlockDropout>({x.i}, p), "block_dropout(const Expression& x, real p)"); }

Expression reshape(const Expression& x, const Dim& d) { return Expression(x.pg, x.pg->add_function<Reshape>({x.i}, d), "reshape(const Expression& x, const Dim& d)"); }
Expression transpose(const Expression& x) { return Expression(x.pg, x.pg->add_function<Transpose>({x.i}), "transpose(const Expression& x)"); }
Expression select_rows(const Expression& x, const vector<unsigned>& rows) { return Expression(x.pg, x.pg->add_function<SelectRows>({x.i}, rows), "select_rows(const Expression& x, const vector<unsigned>& rows)"); }
Expression select_rows(const Expression& x, const vector<unsigned>* prows) { return Expression(x.pg, x.pg->add_function<SelectRows>({x.i}, prows), "select_rows(const Expression& x, const vector<unsigned>* prows)"); }
Expression select_cols(const Expression& x, const vector<unsigned>& cols) { return Expression(x.pg, x.pg->add_function<SelectCols>({x.i}, cols), "select_cols(const Expression& x, const vector<unsigned>& cols)"); }
Expression select_cols(const Expression& x, const vector<unsigned>* pcols) { return Expression(x.pg, x.pg->add_function<SelectCols>({x.i}, pcols), "select_cols(const Expression& x, const vector<unsigned>* pcols)"); }
Expression inverse(const Expression& x) { return Expression(x.pg, x.pg->add_function<MatrixInverse>({x.i}), "inverse(const Expression& x)"); }
Expression logdet(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogDet>({x.i}), "logdet(const Expression& x)"); }

Expression trace_of_product(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<TraceOfProduct>({x.i, y.i}), "trace_of_product(const Expression& x, const Expression& y)");}

Expression squared_norm(const Expression& x) { return Expression(x.pg, x.pg->add_function<SquaredNorm>({x.i}), "squared_norm(const Expression& x)"); }

Expression dot_product(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<DotProduct>({x.i, y.i}), "dot_product(const Expression& x, const Expression& y)"); }
Expression squared_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>({x.i, y.i}), "squared_distance(const Expression& x, const Expression& y)"); }
Expression huber_distance(const Expression& x, const Expression& y, real c) { return Expression(x.pg, x.pg->add_function<HuberDistance>({x.i, y.i}, c), "huber_distance(const Expression& x, const Expression& y, real c)"); }
Expression l1_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<L1Distance>({x.i, y.i}), "l1_distance(const Expression& x, const Expression& y)"); }
Expression binary_log_loss(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>({x.i, y.i}), "binary_log_loss(const Expression& x, const Expression& y)"); }
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m) { return Expression(x.pg, x.pg->add_function<PairwiseRankLoss>({x.i, y.i}, m), "pairwise_rank_loss(const Expression& x, const Expression& y, real m)"); }
Expression poisson_loss(const Expression& x, unsigned y) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, y), "poisson_loss(const Expression& x, unsigned y)"); }
Expression poisson_loss(const Expression& x, const unsigned* py) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, py), "poisson_loss(const Expression& x, const unsigned* py)"); }

Expression conv1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DNarrow>({x.i, f.i}), "conv1d_narrow(const Expression& x, const Expression& f)"); }
Expression conv1d_wide(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DWide>({x.i, f.i}), "conv1d_wide(const Expression& x, const Expression& f)"); }
Expression filter1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Filter1DNarrow>({x.i, f.i}), "filter1d_narrow(const Expression& x, const Expression& f)"); }
Expression kmax_pooling(const Expression& x, unsigned k) { return Expression(x.pg, x.pg->add_function<KMaxPooling>({x.i}, k), "kmax_pooling(const Expression& x, unsigned k)"); }
Expression fold_rows(const Expression& x, unsigned nrows) { return Expression(x.pg, x.pg->add_function<FoldRows>({x.i}, nrows), "fold_rows(const Expression& x, unsigned nrows)"); }

Expression pick(const Expression& x, unsigned v, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v, d), "pick(const Expression& x, unsigned v, unsigned d)"); }
Expression pick(const Expression& x, const vector<unsigned> & v, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v, d), "pick(const Expression& x, const vector<unsigned> & v, unsigned d)"); }
Expression pick(const Expression& x, const unsigned* pv, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv, d), "pick(const Expression& x, const unsigned* pv, unsigned d)"); }
Expression pick(const Expression& x, const vector<unsigned> * pv, unsigned d) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv, d), "pick(const Expression& x, const vector<unsigned> * pv, unsigned d)"); }

Expression pickrange(const Expression& x, unsigned v, unsigned u) { return Expression(x.pg, x.pg->add_function<PickRange>({x.i}, v, u), "pickrange(const Expression& x, unsigned v, unsigned u)"); }

Expression pickneglogsoftmax(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v), "pickneglogsoftmax(const Expression& x, unsigned v)"); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v), "pickneglogsoftmax(const Expression& x, const vector<unsigned> & v)"); }
Expression pickneglogsoftmax(const Expression& x, const unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, pv), "pickneglogsoftmax(const Expression& x, const unsigned* pv)"); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, pv), "pickneglogsoftmax(const Expression& x, const vector<unsigned> * pv)"); }

Expression average_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<AverageColumns>({x.i}), "average_cols(const Expression& x)"); }
Expression sum_dim(const Expression& x, unsigned d) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, d), "sum_dim(const Expression& x, unsigned d)"); }
Expression sum_rows(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, 0), "sum_rows(const Expression& x)"); }
Expression sum_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumDimension>({x.i}, 1), "sum_cols(const Expression& x)"); }

Expression sum_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumBatches>({x.i}), "sum_batches(const Expression& x)"); }

Expression kmh_ngram(const Expression& x, unsigned n) { return Expression(x.pg, x.pg->add_function<KMHNGram>({x.i}, n), "kmh_ngram(const Expression& x, unsigned n)"); }


}
}
