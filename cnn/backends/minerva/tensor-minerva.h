#ifndef CNN_TENSOR_MINERVA_H_
#define CNN_TENSOR_MINERVA_H_

#include <initializer_list>
#include <vector>
#include <memory>
#include <cstring>

#include <Eigen/Eigen>
#include "minerva.h"

using namespace minerva;

namespace cnn {

#define MINERVA_BACKEND 1

typedef minerva::NArray Tensor;
typedef float real;
typedef minerva::Scale Dim;

inline Tensor Zero(const Dim& d) { return minerva::NArray::Zeros(d); }
inline Tensor Random(const Dim& d) { return minerva::NArray::Zeros(d); }
inline Dim size(const Tensor& m) { return m.Size(); }

// avoid using this, because it's slow
inline Tensor FromEigenMatrix(const Eigen::MatrixXf& src) {
  const Dim size = {src.rows(), src.cols()};
  std::shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  auto p = src.data();
  auto d = data.get();
  std::memcpy(d, p, size.Prod() * sizeof(float));
  return NArray::MakeNArray(size, data);
}

// in column-major order, consecutive elements of the columns are contiguous.
// in Minerva, matrices are stored in column-major (i.e., FORTRAN) order
inline Tensor Ccm(const Dim&d, const std::initializer_list<real>& v) {
  std::vector<real> vv = v;
  std::shared_ptr<float> input_ptr(new float[d.Prod()], [](float* ptr) { delete[] ptr; });
  std::memcpy(input_ptr.get(), &vv[0], d.Prod() * sizeof(float));
  return minerva::NArray::MakeNArray(d, input_ptr);
}

} // namespace cnn

#endif
