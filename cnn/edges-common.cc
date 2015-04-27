#include "cnn/edges.h"

#include <limits>
#include <cmath>
#include <sstream>

using namespace std;

namespace cnn {

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inner(" << arg_names[0] << "," << arg_names[1] << ") + " << arg_names[2];
  return s.str();
}

string GaussianNoise::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " + N(0," << stddev << ')';
  return s.str();
}

string Dropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

string OneMinusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "1 - " << arg_names[0];
  return s.str();
}

string Sum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < tail.size(); ++i)
    s << " + " << arg_names[i];
  return s.str();
}

string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

string Square::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "square(" << arg_names[0] << ')';
  return s.str();
}

string Exp::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "exp(" << arg_names[0] << ')';
  return os.str();
}

string Log::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "log(" << arg_names[0] << ')';
  return os.str();
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

string ConcatenateColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat_cols(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

string Hinge::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "hinge(" << arg_names[0] << ",m=" << margin << ")";
  return os.str();
}

string Identity::as_string(const vector<string>& arg_names) const {
  return arg_names[0];
}

string MaxPooling1D::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "maxpool1d(" << arg_names.front() << ",w=" << width << ")";
  return os.str();
}

string Softmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softmax(" << arg_names[0] << ')';
  return s.str();
}

string LogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ')';
  return s.str();
}

string RestrictedLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "r_log_softmax(" << arg_names[0] << ')';
  return s.str();
}

string PickElement::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick(" << arg_names[0] << ',' << *pval << ')';
  return s.str();
}

// x_1 is a vector
// y = (x_1)[start:end]
string PickRange::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "slice(" << arg_names[0] << ',' << start << ':' << end << ')';
  return s.str();
}

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

string CwiseMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " \\cdot " << arg_names[1];
  return s.str();
}

string Multilinear::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

string Negate::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << '-' << arg_names[0];
  return s.str();
}

string Rectify::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ReLU(" << arg_names[0] << ')';
  return s.str();
}

string SquaredEuclideanDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||^2";
  return s.str();
}

string LogisticSigmoid::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "\\sigma(" << arg_names[0] << ')';
  return s.str();
}

string BinaryLogLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "binary_log_loss(" << arg_names[0] << ", " << *ptarget_y << ')';
  return os.str();
}

} // namespace cnn
