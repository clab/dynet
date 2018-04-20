#ifndef _H_TOPK
#define _H_TOPK

//#define HAVE_CUDA 1

#ifdef __CUDACC__

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

// using eigen, but only one-argmax
namespace impl_eigen{
  template<typename T, typename IndexType, bool IsMax=true>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k,
    Eigen::GpuDevice& my_device){
    throw std::runtime_error("Not implemented topk-EIGEN for IndexType != Eigen::DenseIndex.");
  }

  template<typename T, bool IsMax=true>
  cudaError topk(T* input, T* ouput_val, Eigen::DenseIndex* ouput_idx,
    Eigen::DenseIndex outer_size, Eigen::DenseIndex inner_size, Eigen::DenseIndex cur_size, Eigen::DenseIndex k,
    Eigen::GpuDevice& my_device){
    // only argmax
    if(k != 1)
      throw std::runtime_error("Not implemented topk-EIGEN for k != 1.");
    const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
    if(IsMax){
      (Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(ouput_idx, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).argmax(1);
      (Eigen::TensorMap<Eigen::Tensor<T, 2>>(ouput_val, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).maximum(reduction_axis);
    }
    else{
      (Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(ouput_idx, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).argmin(1);
      (Eigen::TensorMap<Eigen::Tensor<T, 2>>(ouput_val, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).minimum(reduction_axis);
    }
    return cudaGetLastError();
  }
};

#include "topk_tf.h"
#include "topk_tr.h"

namespace topk_gpu{

  enum TopK_Gpu_Strategy { TOPK_AUTO=0, TOPK_TF, TOPK_TR, TOPK_EIGEN };

  template<typename T, typename IndexType, bool IsMax, int Strategy>
  cudaError topk(T* input, T* output_val, IndexType* output_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k, Eigen::GpuDevice* dev=nullptr){
    // auto mode depending on k
    auto strat = Strategy;
    if(Strategy == TOPK_AUTO){
      // TODO: simple rule based (can be further extended)
      strat = ((k<=16) ? TOPK_TF : TOPK_TR);
    }
    if(strat == TOPK_TF)   // max-shards
      return impl_tf::topk<T, IndexType, IsMax>(input, output_val, output_idx, outer_size, inner_size, cur_size, k, 0);
    else if(strat == TOPK_TR)
      return impl_tr::topk<T, IndexType, IsMax>(input, output_val, output_idx, outer_size, inner_size, cur_size, k);
    else if(strat == TOPK_EIGEN)
      return impl_eigen::topk<T, IndexType, IsMax>(input, output_val, output_idx, outer_size, inner_size, cur_size, k, *dev);
    else
      throw std::runtime_error("Unknown topk strategy.");
  }
}

#endif

#include <algorithm>
#include <vector>

// simple loop with nth-element
namespace topk_cpu{
  template<typename T>
  bool gt(const T& a, const T& b){ return a.value > b.value;}

  template<typename T>
  bool lt(const T& a, const T& b){ return a.value < b.value; }

  template <typename T, typename IndexType>
  struct Entry {
    IndexType index;
    T value;
  };

  template<typename T, typename IndexType, bool IsMax>
  void topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k){
    using std::vector;
    for(IndexType i = 0; i < outer_size*inner_size; i++){
      IndexType slice = i;
      T* input_start = input + slice/inner_size*inner_size*cur_size + slice%inner_size;
      T* output_val_start = ouput_val + slice/inner_size*inner_size*k + slice%inner_size;
      IndexType* ouput_idx_start = ouput_idx + slice/inner_size*inner_size*k + slice%inner_size;
      // copy into a vector
      vector<Entry<T, IndexType>> entries;
      for(IndexType step = 0; step < cur_size; step++)
        entries.push_back({step, input_start[step*inner_size]});
      // split
      if(IsMax)
        std::nth_element(entries.begin(), entries.begin()+k-1, entries.end(), gt<Entry<T, IndexType>>);
      else
        std::nth_element(entries.begin(), entries.begin()+k-1, entries.end(), lt<Entry<T, IndexType>>);
      // assign
      for(IndexType step = 0; step < k; step++){
        const auto& one = entries[step];
        output_val_start[step*inner_size] = one.value;
        ouput_idx_start[step*inner_size] = one.index;
      }
    }
  }
}

#endif // !_H_TOPK

