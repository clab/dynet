#ifndef CNN_TEST_UTILS_H_
#define CNN_TEST_UTILS_H_

#include "cnn/tensor.h"

namespace cnn {

#if WITH_MINERVA_BACKEND

struct TestTensorSetup {
  TestTensorSetup() {
    int argc = 1;
    char* foo = "foo";
    char** argv = {&foo};
    minerva::MinervaSystem::Initialize(&argc, &argv);
#if HAS_CUDA
    minerva::MinervaSystem::Instance().device_manager().CreateGpuDevice(0);
#else
    minerva::MinervaSystem::Instance().device_manager().CreateCpuDevice();
#endif
  }
};

double t(const Tensor& T, unsigned i, unsigned j) {
  int m = T.Size(0);
  return T.Get().get()[j * m + i];
}

std::ostream& operator<<(std::ostream& os, const Tensor& T) {
  if (T.Size().NumDims() == 2) {
    int m = T.Size(0);
    int n = T.Size(1);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        os << '\t' << t(T,i,j);
      }
      os << std::endl;
    }
    return os;
  } else {
    os << T.Size() << ": ";
    minerva::FileFormat ff; ff.binary = false;
    T.ToStream(os, ff);
    return os;
  }
}

#else

struct TestTensorSetup {
  TestTensorSetup() {
    int argc = 1;
    char* p = "foo";
    char** argv = {&p};
    cnn::Initialize(argc, argv);
  }
};

double t(const Tensor& T, unsigned i, unsigned j) {
#if WITH_THPP_BACKEND
  return T.at({i,j});
#else
  return T(i, j);
#endif
}

double t(const Tensor& T, unsigned i) {
#if WITH_THPP_BACKEND
  return T.at({i});
#else
  return T(i, 0);
#endif
}

#endif

} // namespace cnn

#endif
