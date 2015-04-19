#ifndef CNN_THPP_TENSOR_H_
#define CNN_THPP_TENSOR_H_

#include <initializer_list>

#include <Eigen/Eigen>
#include "thpp/Tensor.h"
#include "cnn/backends/thpp/random.h"
#include "cnn/backends/thpp/dim.h"

#define THPP_BACKEND 1

namespace cnn {

void Initialize(int& argc, char**& argv);

typedef thpp::Tensor<float> Tensor;
typedef float real;

inline real as_scalar(const Tensor& t) {
  assert(t.isScalar());
  return t.front();
}

inline std::vector<real> as_vector(const Tensor& t) {
  std::vector<real> res(t.size());
  std::memcpy(&res[0], t.data(), sizeof(real) * t.size());
  return res;
}

inline Tensor Constant(const Dim& d, real c) {
  Tensor t(d);
  t.fill(c);
  return t;
}
inline Tensor Zero(const Dim& d) {
  Tensor z(d);
  z.zero();
  return z;
}

inline Dim size(const Tensor& m) { return m.sizes(); }

inline size_t num_params(const Tensor& m) { return m.size(); }

inline real rand01() {
  std::uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

// avoid using this, because it's slow
inline Tensor FromEigenMatrix(const Eigen::MatrixXf& src) {
  if (src.cols() == 1) {
    Tensor t({src.rows()});
    auto p = t.storage();
    int i = 0;
    for (int r = 0; r < src.rows(); ++r)
      p[i++] = src(r, 0);
    return t;
  } else {
    Tensor t({src.rows(), src.cols()});
    auto p = t.storage();
    int i = 0;
    for (int r = 0; r < src.rows(); ++r)
      for (int c = 0; c < src.cols(); ++c)
        p[i++] = src(r,c);
    return t;
  }
}

inline Tensor FromRawData(const Dim& dim, const float* data) {
  Tensor t(dim);
  std::memcpy(t.data(), data, sizeof(float) * dim.size());
  return t;
}

inline Tensor Random(const Dim& d, real scale) {
  // TODO replace with TH-appropriate thing
  std::uniform_real_distribution<real> distribution(-scale,scale);
  auto b = [&] (real) {return distribution(*rndeng);};
  int cols = 1;
  if (d.ndims() > 1) cols = d.size(1);
  if (d.ndims() > 2) { assert(!"not implemented"); }
  return FromEigenMatrix(Eigen::MatrixXf::NullaryExpr(d.size(0), cols, b));
}

inline Tensor Random(const Dim& d) {
  return Random(d, sqrt(6) / sqrt(d.size()));
}

inline Tensor RandomBernoulli(const Dim& d, real p) {
  // TODO replace with TH-appropriate thing
  std::bernoulli_distribution distribution(p);
  auto b = [&] (real) {return distribution(*rndeng);};
  int cols = 1;
  if (d.ndims() > 1) cols = d.size(1);
  if (d.ndims() > 2) { assert(!"not implemented"); }
  return FromEigenMatrix(Eigen::MatrixXf::NullaryExpr(d.size(0), cols, b));
}

inline Tensor RandomNormal(const Dim& d, real mean, real stddev) {
  // TODO replace with TH-appropriate thing
  std::normal_distribution<real> distribution(mean, stddev);
  auto b = [&] (real) {return distribution(*rndeng);};
  int cols = 1;
  if (d.ndims() > 1) cols = d.size(1);
  if (d.ndims() > 2) { assert(!"not implemented"); }
  return FromEigenMatrix(Eigen::MatrixXf::NullaryExpr(d.size(0), cols, b));
}

inline Tensor Crm(std::initializer_list<long> dim, const std::initializer_list<real>& v) {
  Tensor t(dim);
  auto p = t.storage();
  int i = 0;
  for (const auto& x : v) {
    p[i++] = x;
  }
  return t;
}

// in column-major order, consecutive elements of the columns are contiguous.
// in TH, matrices are stored in row-major (i.e., C) order
inline Tensor Ccm(std::initializer_list<long> dim, const std::initializer_list<real>& v) {
  if (dim.size() == 1) return Crm(dim, v);
  assert(dim.size() == 2);
  Tensor t(dim);
  t.zero();
  int cc = 0;
  int cr = 0;
  for (const auto& x : v) {
    t.at({cr, cc}) = x;
    ++cr;
    if (cr == t.size(0)) { cr = 0; ++cc; }
  }
  return t;
}

inline std::string str(const Tensor& T) {
  std::ostringstream os;
  if (T.ndims() == 2) {
    int m = T.size(0);
    int n = T.size(1);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        os << '\t' << T.at({i,j});
      }
      os << std::endl;
    }
  } else if (T.ndims() == 1) {
    os << T << std::endl;
    for (int i = 0; i < T.size(); ++i) { os << '\t' << T.at(i) << std::endl; }
  } else {
    os << T;
    for (int i = 0; i < T.size(); ++i) { os << ' ' << T.at(i); }
    os << std::endl;
  }
  return os.str();
}

} // namespace cnn

#endif
