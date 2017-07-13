#ifndef DYNET_CUDNN_TYPES_H_
#define DYNET_CUDNN_TYPES_H_

#include "dynet/dynet.h"
#include "dynet/cuda.h"

#if HAVE_CUDNN
template <class T>
struct DataTypeToCudnnType {};

#define MATCH_TYPE_TO_CUDNN_TYPE(TYPE, ENUM)   \
  template <>                                  \
  struct DataTypeToCudnnType<TYPE> {           \
    static const cudnnDataType_t value = ENUM; \
  }

MATCH_TYPE_TO_CUDNN_TYPE(float, CUDNN_DATA_FLOAT);
MATCH_TYPE_TO_CUDNN_TYPE(double, CUDNN_DATA_DOUBLE);

#undef MATCH_TYPE_TO_CUDNN_TYPE
#endif

namespace dynet {

// A helper class to allocate memory from the aux_mem pointer for complex operators
// e.g. Conv2D
struct NodeMemPool {
 public:
  explicit NodeMemPool() : capacity_(0), used_(0), mem_(NULL) {}
  explicit NodeMemPool(const int capacity, void* mem) 
      : capacity_(capacity), used_(0), mem_(mem) {}

  void* allocate(size_t nbytes) {
    if (used_ + nbytes > capacity_) {
      std::ostringstream oss; oss  
          << "aux_mem_pool allocate memory failed: exceed maximally allowed size";
      throw std::runtime_error(oss.str());
    }
    void* res = static_cast<char*>(mem_) + used_;
    used_ += nbytes;
    return res;
  }

  void free() {
    used_ = 0;
  }

  void* head() {
    return mem_;
  }

  size_t size() {
    return capacity_;
  }

  size_t used() {
    return used_;
  }

 private:
  size_t capacity_;
  size_t used_;
  void* mem_;
};

/* layout transformation functionsi (ColMajor) */
//shuffle is not supported by ThreadPool device
template<class MyDevice>
struct HWCNToCHWN {
  void operator()(const Tensor* in, Tensor& out, const MyDevice & dev) {
    if (in->device->type == DeviceType::ThreadPool) {
      const unsigned N = in->d.nd == 4 ? in->d[3] : in->d.bd; 
      const unsigned C = in->d[2];
      const unsigned H = in->d[0];
      const unsigned W = in->d[1];
      for (unsigned n = 0; n < N; ++n)
        for (unsigned c = 0; c < C; ++c)
          for (unsigned h = 0; h < H; ++h)
            for (unsigned w = 0; w < W; ++w)
              out.v[n*H*W*C+w*H*C+h*C+c] = in->v[n*H*W*C+c*H*W+w*H+h];
    } else {
      Eigen::array<ptrdiff_t, 4> shuffles = {2, 0, 1, 3};
      //shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
      if (in->d.nd == 4) {
        out.t<4>().device(*dev.edevice) = in->t<4>().shuffle(shuffles);
      } else {
        out.tb<3>().device(*dev.edevice) = in->tb<3>().shuffle(shuffles);
      }
    }
  }
};

template<class MyDevice>
struct HWCNToNCHW {
  void operator()(const Tensor* in, Tensor& out, const MyDevice & dev) {
    if (in->device->type == DeviceType::ThreadPool) {
      const unsigned N = in->d.nd == 4 ? in->d[3] : in->d.bd; 
      const unsigned C = in->d[2];
      const unsigned H = in->d[0];
      const unsigned W = in->d[1];
      for (unsigned n = 0; n < N; ++n)
        for (unsigned c = 0; c < C; ++c)
          for (unsigned h = 0; h < H; ++h)
            for (unsigned w = 0; w < W; ++w)
              out.v[w*H*C*N+h*C*N+c*N+n] = in->v[n*H*W*C+c*H*W+w*H+h];
    } else {
      Eigen::array<ptrdiff_t, 4> shuffles = {3, 2, 0, 1};
      //shuffles[0] = 2; shuffles[1] = 0; shuffles[2] = 1; shuffles[3] = 3;
      if (in->d.nd == 4) {
        out.t<4>().device(*dev.edevice) = in->t<4>().shuffle(shuffles);
      } else {
        out.tb<3>().device(*dev.edevice) = in->tb<3>().shuffle(shuffles);
      }
    }
  }
};

template<class MyDevice>
struct CHWNToHWCN {
  void operator()(const Tensor* in, Tensor& out, const MyDevice & dev) {
    if (in->device->type == DeviceType::ThreadPool) {
      const unsigned N = in->d.nd == 4 ? in->d[3] : in->d.bd; 
      const unsigned C = in->d[0];
      const unsigned H = in->d[1];
      const unsigned W = in->d[2];
      for (unsigned n = 0; n < N; ++n)
        for (unsigned c = 0; c < C; ++c)
          for (unsigned h = 0; h < H; ++h)
            for (unsigned w = 0; w < W; ++w)
              out.v[n*H*W*C+c*H*W+w*H+h] = in->v[n*H*W*C+w*H*C+h*C+c];
    } else {
      Eigen::array<ptrdiff_t, 4> shuffles = {1, 2, 0, 3};
      if (in->d.nd == 4) {
        out.t<4>().device(*dev.edevice) = in->t<4>().shuffle(shuffles);
      } else {
        out.tb<3>().device(*dev.edevice) = in->tb<3>().shuffle(shuffles);
      }
    }
  }
};

template<class MyDevice>
struct NCHWToHWCN {
  void operator()(const Tensor* in, Tensor& out, const MyDevice & dev) {
    if (in->device->type == DeviceType::ThreadPool) {
      const unsigned N = in->d[0];
      const unsigned C = in->d[1];
      const unsigned H = in->d[2];
      const unsigned W = in->d.bd == 4 ? in->d[3] : in->d.bd;
      for (unsigned n = 0; n < N; ++n)
        for (unsigned c = 0; c < C; ++c)
          for (unsigned h = 0; h < H; ++h)
            for (unsigned w = 0; w < W; ++w)
              out.v[n*H*W*C+c*H*W+w*H+h] = in->v[w*H*C*N+h*C*N+c*N+n];
    } else {
      Eigen::array<ptrdiff_t, 4> shuffles = {2, 3, 1, 0};
      if (in->d.nd == 4) {
        out.t<4>().device(*dev.edevice) = in->t<4>().shuffle(shuffles);
      } else {
        out.tb<3>().device(*dev.edevice) = in->tb<3>().shuffle(shuffles);
      }
    }
  }
};

} // namespace dynet

#endif
