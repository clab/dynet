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

 private:
  size_t capacity_;
  size_t used_;
  void* mem_;
};

// template?
struct NCHWToNHWC {
  void operator()(const Tensor* in, Tensor& out) {
#if HAVE_CUDA
    throw std::runtime_error("Tensor::NCHWToNHWC not implemented for CUDA");
#else
    const unsigned N = (out.d.nd == 4 ? out.d[3]:out.d.bd);
    const unsigned C = out.d[0];
    const unsigned H = out.d[2];
    const unsigned W = out.d[1];
    for (unsigned n = 0; n < N; ++n)
      for (unsigned c = 0; c < C; ++c) 
        for (unsigned h = 0; h < H; ++h)
          for (unsigned w = 0; w < W; ++w)
            out.v[n*H*W*C+h*W*C+w*C+c] = in->v[n*H*W*C+c*H*W+h*W+w];
#endif
  }
};

struct NHWCToNCHW {
  void operator()(const Tensor* in, Tensor& out) {
#if HAVE_CUDA
    throw std::runtime_error("Tensor::NHWCToNCHW not implemented for CUDA");
#else
    const unsigned N = (out.d.nd == 4 ? out.d[3]:out.d.bd);
    const unsigned C = out.d[2];
    const unsigned H = out.d[1];
    const unsigned W = out.d[0];
    for (unsigned n = 0; n < N; ++n)
      for (unsigned c = 0; c < C; ++c) 
        for (unsigned h = 0; h < H; ++h)
          for (unsigned w = 0; w < W; ++w)
            out.v[n*H*W*C+c*W*H+h*W+w] = in->v[n*H*W*C+h*C*W+w*C+c];
#endif
  } 
};

struct NCHWToWHCN {
  void operator()(const Tensor* in, Tensor& out) {
#if HAVE_CUDA
    throw std::runtime_error("Tensor::NCHWToWHCN not implemented for CUDA");
#else
    const unsigned N = out.d[0]; //only applies on filters
    const unsigned C = out.d[1];
    const unsigned H = out.d[2];
    const unsigned W = out.d[3];
    for (unsigned n = 0; n < N; ++n)
      for (unsigned c = 0; c < C; ++c)
        for (unsigned h = 0; h < H; ++h)
          for (unsigned w = 0; w < W; ++w)
            out.v[w*H*C*N+h*C*N+c*N+n] = in->v[n*H*C*W+c*H*W+h*W+w];
#endif
  }
};

struct WHCNToNCHW {
  void operator()(const Tensor* in, Tensor& out) {
#if HAVE_CUDA
    throw std::runtime_error("Tensor::WHCNToNCHW not implemented for CUDA");
#else
    const unsigned N = out.d[3];
    const unsigned C = out.d[2];
    const unsigned H = out.d[1];
    const unsigned W = out.d[0];
    for (unsigned n = 0; n < N; ++n)
      for (unsigned c = 0; c < C; ++c)
        for (unsigned h = 0; h < H; ++h)
          for (unsigned w = 0; w < W; ++w)
            out.v[n*C*H*W+c*H*W+h*W+w] = in->v[w*H*C*N+h*C*N+c*N+n];
#endif
  }
};

} // namespace dynet

#endif
