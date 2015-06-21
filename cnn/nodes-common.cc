#include "cnn/nodes.h"

#include <limits>
#include <cmath>
#include <sstream>

using namespace std;

namespace cnn {

inline bool LooksLikeVector(const Dim& d) {
  if (d.ndims() == 1) return true;
  if (d.ndims() > 1) {
    for (int i = 1; i < d.ndims(); ++i)
      if (d[i] != 1) return false;
  }
  return true;
}

string ConstScalarMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << alpha;
  return s.str();
}

Dim ConstScalarMultiply::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1) {
    cerr << "ConstScalarMultiply expects one argument: " << xs << endl;
    abort();
  }
  return xs[0];
}

string DotProduct::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << "^T . " << arg_names[1];
  return s.str();
}

Dim DotProduct::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  assert(LooksLikeVector(xs[0]));
  assert(LooksLikeVector(xs[1]));
  assert(xs[0].rows() == xs[1].rows());
  return Dim({1});
}

string Transpose::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << "^T";
  return s.str();
}

Dim Transpose::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0].transpose();
}

string Reshape::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "reshape(" << arg_names[0] << " --> " << to << ')';
  return s.str();
}

Dim Reshape::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  assert(xs[0].size() == to.size());
  return to;
}

string SumColumns::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_cols(matrix=" << arg_names[0];
  if (arg_names.size() == 2) s << ", col_weighting=" << arg_names[1];
  s << ')';
  return s.str();
}

Dim SumColumns::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1 || xs.size() == 2);
  return Dim({xs[0].rows()});
}

string KMHNGram::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "kmh-ngram(" << arg_names[0] << ')';
  return s.str();
}

Dim KMHNGram::dim_forward(const vector<Dim>& xs) const {
  assert(xs[0].ndims() == 2);
  const int new_cols = xs[0].cols() - n + 1;
  if (new_cols < 1) {
    cerr << "Bad input dimensions in KMHNGram: " << xs << endl;
    abort();
  }
  return Dim({xs[0][0], new_cols});
}

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dot(" << arg_names[0] << "," << arg_names[1] << ')';
  if (arg_names.size() == 3) s << " + " << arg_names[2];
  return s.str();
}

Dim InnerProduct3D_1D::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 && xs.size() != 3) {
    cerr << "Expected two or three arguments in InnerProduct3D_1D\n";
    abort();
  }
  if (xs[0].ndims() != 3 ||
      xs[1].ndims() != 1 ||
      xs[0].size(2) != xs[1].size(0)) {
    cerr << "Bad input dimensions in InnerProduct3D_1D: " << xs << endl;
    abort();
  }
  Dim d({xs[0].size(0), xs[0].size(1)});
  if (xs.size() == 3 && xs[2] != d) {
    cerr << "Bad input dimensions in InnerProduct3D_1D: " << xs << endl;
    abort();
  }
  return d;
}

string GaussianNoise::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " + N(0," << stddev << ')';
  return s.str();
}

Dim GaussianNoise::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Dropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

Dim Dropout::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string ConstantMinusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << c << " - " << arg_names[0];
  return s.str();
}

Dim ConstantMinusX::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Sum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + " << arg_names[i];
  return s.str();
}

Dim Sum::dim_forward(const vector<Dim>& xs) const {
  for (unsigned i = 1; i < xs.size(); ++i) {
    if (xs[0] != xs[1]) {
      cerr << "Mismatched input dimensions in Sum: " << xs << endl;
      abort();
    }
  }
  return xs[0];
}

string Average::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "average(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << ", " << arg_names[i];
  s << ")";
  return s.str();
}

Dim Average::dim_forward(const vector<Dim>& xs) const {
  for (unsigned i = 1; i < xs.size(); ++i) {
    if (xs[0] != xs[1]) {
      cerr << "Mismatched input dimensions in Average: " << xs << endl;
      abort();
    }
  }
  return xs[0];
}

string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

Dim Tanh::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Square::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "square(" << arg_names[0] << ')';
  return s.str();
}

Dim Square::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Exp::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "exp(" << arg_names[0] << ')';
  return os.str();
}

Dim Exp::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Log::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "log(" << arg_names[0] << ')';
  return os.str();
}

Dim Log::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Concatenate::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Dim Concatenate::dim_forward(const vector<Dim>& xs) const {
  unsigned new_rows = 0;
  for (auto& d : xs) {
    if (!LooksLikeVector(d)) {
      cerr << "Bad input dimensions in Concatenate: " << xs << endl;
      abort();
    }
    new_rows += d[0];
  }
  return Dim({new_rows});
}

string ConcatenateColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat_cols(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Dim ConcatenateColumns::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() > 0);
  int rows = xs[0][0];
  int new_cols = 0;
  for (auto& d : xs) {
    if (d[0] != rows) {
      cerr << "Bad input dimensions in ConcatenateColumns: " << xs << endl;
      abort();
    }
    new_cols += d[1];
  }
  return Dim({rows, new_cols});
}

string PairwiseRankLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "max(0, " << margin << " - " << arg_names[0] << " + " << arg_names[1] << ')';
  return os.str();
}

Dim PairwiseRankLoss::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 ||
      xs[0] != xs[1] ||
      xs[0].rows() != 1 ||
      (xs[0].ndims() != 1 && xs[0].ndims() != 2)) {
    cerr << "Bad input dimensions in PairwiseRankLoss: " << xs << endl;
    abort();
  }
  return xs[0];
}

string Hinge::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "hinge(" << arg_names[0] << ", pe=" << pelement << ", m=" << margin << ')';
  return os.str();
}

Dim Hinge::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 1 || !LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in Hinge: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string Identity::as_string(const vector<string>& arg_names) const {
  return arg_names[0];
}

Dim Identity::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string MaxPooling1D::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "maxpool1d(" << arg_names.front() << ",w=" << width << ")";
  return os.str();
}

Dim MaxPooling1D::dim_forward(const vector<Dim>& xs) const {
  cerr << "MaxPooling1D::dim_forward not implemented\n";
  abort();
}

string Softmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim Softmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in Softmax: " << xs << endl;
    abort();
  }
  return xs[0];
}

string SoftSign::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softsign(" << arg_names[0] << ')';
  return s.str();
}

Dim SoftSign::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in Softsign: " << xs << endl;
    abort();
  }
  return xs[0];
}

string PickNegLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ")_{" << *pval << '}';
  return s.str();
}

Dim PickNegLogSoftmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in PickNegLogSoftmax: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string LogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim LogSoftmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in LogSoftmax: " << xs << endl;
    abort();
  }
  return xs[0];
}

string RestrictedLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "r_log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim RestrictedLogSoftmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in RestrictedLogSoftmax: " << xs << endl;
    abort();
  }
  return xs[0];
}

string PickElement::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick(" << arg_names[0] << ',' << *pval << ')';
  return s.str();
}

Dim PickElement::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in PickElement: " << xs << endl;
    abort();
  }
  return Dim({1});
}

// x_1 is a vector
// y = (x_1)[start:end]
string PickRange::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "slice(" << arg_names[0] << ',' << start << ':' << end << ')';
  return s.str();
}

Dim PickRange::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in PickElement: " << xs << endl;
    abort();
  }
  assert((int)end <= xs[0][0]);
  return Dim({end - start});
}

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Dim MatrixMultiply::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0].cols() != xs[1].rows()) {
    cerr << "Mismatched input dimensions in MatrixMultiply: " << xs << endl;
    abort();
  }
  if (xs[1].ndims() == 1) return Dim({xs[0].rows()});
  return Dim({xs[0].rows(), xs[1].cols()});
}

string CwiseMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " \\cdot " << arg_names[1];
  return s.str();
}

Dim CwiseMultiply::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in CwiseMultiply: " << xs << endl;
    abort();
  }
  return xs[0];
}

string CwiseQuotient::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " / " << arg_names[1];
  return s.str();
}

Dim CwiseQuotient::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in CwiseQuotient: " << xs << endl;
    abort();
  }
  return xs[0];
}

string AffineTransform::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

Dim AffineTransform::dim_forward(const vector<Dim>& xs) const {
  if ((xs.size() - 1) % 2 != 0) {
    cerr << "Bad number of inputs for AffineTransform: " << xs << endl;
    abort();
  }
  for (unsigned i = 1; i < xs.size(); i += 2) {
    if (xs[i].cols() != xs[i+1].rows() ||
        xs[0].rows() != xs[i].rows() ||
        xs[0].cols() != xs[i+1].cols()) {
      cerr << "Bad dimensions for AffineTransform: " << xs << endl;
      abort();
    }
  }
  return xs[0];
}

string Negate::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << '-' << arg_names[0];
  return s.str();
}

Dim Negate::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Rectify::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ReLU(" << arg_names[0] << ')';
  return s.str();
}

Dim Rectify::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string HuberDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||_H(" << d << ')';
  return s.str();
}

Dim HuberDistance::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in HuberDistance: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string L1Distance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||_1";
  return s.str();
}

Dim L1Distance::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in L1Distance: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string SquaredEuclideanDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||^2";
  return s.str();
}

Dim SquaredEuclideanDistance::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in SquaredEuclideanDistance: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string LogisticSigmoid::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "\\sigma(" << arg_names[0] << ')';
  return s.str();
}

Dim LogisticSigmoid::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string BinaryLogLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "binary_log_loss(" << arg_names[0] << ", " << arg_names[1] << ')';
  return os.str();
}

Dim BinaryLogLoss::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0].rows() != 2 && xs[0].ndims() != 1) {
    cerr << "Bad input dimensions in BinaryLogLoss: " << xs << endl;
    abort();
  }
  if (xs[1].rows() != 2 && xs[1].ndims() != 1) {
    cerr << "Bad input dimensions in BinaryLogLoss: " << xs << endl;
    abort();
  }
  return Dim({1});
}

} // namespace cnn
