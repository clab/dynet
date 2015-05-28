#include "cnn/expr.h"

namespace cnn { namespace expr {

Expression input(ComputationGraph& g, real s) { return Expression(&g, g.add_input(s)); }
Expression input(ComputationGraph& g, const real *ps) { return Expression(&g, g.add_input(ps)); }
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>({x.i})); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Sum>({x.i, y.i})); }
Expression operator-(const Expression& x, const Expression& y) { return x+(-y); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>({x.i, y.i})); }

Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>({x.i})); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>({x.i})); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>({x.i})); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>({x.i})); }

Expression squaredDistance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>({x.i, y.i})); }
Expression binary_log_loss(const Expression& x, real ty) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>({x.i}, &ty)); }
Expression binary_log_loss(const Expression& x, real* pty) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>({x.i}, pty)); }
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m) { return Expression(x.pg, x.pg->add_function<PairwiseRankLoss>({x.i, y.i}, m)); }

Expression pick(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v)); }

Expression sum_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumColumns>({x.i})); }

Expression kmh_ngram(const Expression& x, unsigned n) { return Expression(x.pg, x.pg->add_function<KMHNGram>({x.i}, n)); }

} }
