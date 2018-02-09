#ifndef DYNET_INDEX_TENSOR_H
#define DYNET_INDEX_TENSOR_H

#include "dynet/tensor.h"

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include <unsupported/Eigen/CXX11/Tensor>

namespace dynet {

/**
 * \ingroup tensor
 * \brief Represents a tensor of indices
 * \details This holds indices to locations within a dimension or tensor.
 *
 */
struct IndexTensor {
  /**
   * \brief Create an empty tensor
   */
  IndexTensor() : d(Dim()), v(nullptr), device(nullptr), mem_pool(DeviceMempool::NONE) { }
  /**
   * \brief Creates a tensor
   * \details [long description]
   *
   * \param d Shape of the tensor
   * \param v Pointer to the values
   * \param dev Device
   * \param mem Memory pool
   */
  IndexTensor(const Dim& d, Eigen::DenseIndex* v, Device* dev, DeviceMempool mem) : d(d), v(v), device(dev), mem_pool(mem) {}

  Dim d;  /**< Shape of tensor */
  Eigen::DenseIndex* v;  /**< Pointer to memory */
  Device* device;
  DeviceMempool mem_pool;

};

// Get view as an Eigen Tensor (see specializations below-- this is to work Eigen's and DyNet's compile-type vs. run-time differences)
/**
 * \brief Get view as a Tensor
 * \tparam Order Tensor order. Order 0 through 4 are already implemented for you
 * \return Eigen Tensor of the given order
 */
template <int Order> Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, Order>> t(IndexTensor & t);
template <int Order> const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, Order>> t(const IndexTensor & t);

template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 0>> t<0>(IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.size() == 1, "Illegal access of tensor in function t<0>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 0>>(t.v);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 0>> t<0>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.size() == 1, "Illegal access of tensor in function t<0>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 0>>(t.v);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>> t<1>(IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && (t.d.ndims() == 1 || t.d.size() == t.d.rows()), "Illegal access of tensor in function t<1>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>>(t.v, (int)t.d[0]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>> t<1>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && (t.d.ndims() == 1 || t.d.size() == t.d.rows()), "Illegal access of tensor in function t<1>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>>(t.v, (int)t.d[0]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>> t<2>(IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 2, "Illegal access of tensor in function t<2>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(t.v, (int)t.d[0], (int)t.d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(t.v, (int)t.d[0], (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>> t<2>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 2, "Illegal access of tensor in function t<2>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(t.v, (int)t.d[0], (int)t.d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(t.v, (int)t.d[0], (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> t<3>(IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 3, "Illegal access of tensor in function t<3>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2]);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> t<3>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 3, "Illegal access of tensor in function t<3>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2]);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)1, (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>> t<4>(IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 4, "Illegal access of tensor in function t<4>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3]);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>> t<4>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 4, "Illegal access of tensor in function t<4>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3]);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1);
}
// ...

/**
 * \brief Get view as an Eigen Tensor where the final dimension is the various batches
 * \tparam Order Tensor order. Order 0 through 4 are already implemented for you
 * \return Eigen Tensor of the given order + 1
 */
template <int Order> Eigen::TensorMap < Eigen::Tensor < Eigen::DenseIndex, Order + 1 >> tb(IndexTensor & t);
template <int Order> const Eigen::TensorMap < Eigen::Tensor < Eigen::DenseIndex, Order + 1 >> tb(const IndexTensor & t);

template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>> tb<0>(IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_size() == 1, "Illegal access of tensor in function tb<0>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>>(t.v, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>> tb<0>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.batch_size() == 1, "Illegal access of tensor in function tb<0>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 1>>(t.v, (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>> tb<1>(IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() == 1 || t.d.batch_size() == t.d.rows(), "Illegal access of tensor in function tb<1>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(t.v, (int)t.d[0], (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>> tb<1>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() == 1 || t.d.batch_size() == t.d.rows(), "Illegal access of tensor in function tb<1>(IndexTensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 2>>(t.v, (int)t.d[0], (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> tb<2>(IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 2, "Illegal access of tensor in function tb<2>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)1, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>> tb<2>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 2, "Illegal access of tensor in function tb<2>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 3>>(t.v, (int)t.d[0], (int)1, (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>> tb<3>(IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 3, "Illegal access of tensor in function tb<3>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>> tb<3>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 3, "Illegal access of tensor in function tb<3>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>> tb<4>(IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 4, "Illegal access of tensor in function tb<4>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3], (int)t.d.bd);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1, (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>> tb<4>(const IndexTensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 4, "Illegal access of tensor in function tb<4>(IndexTensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3], (int)t.d.bd);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1, (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<Eigen::DenseIndex, 5>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1, (int)t.d.bd);
}
// ...

/**
 * \ingroup tensor
 * \brief Get the array of indices in an index tensor
 * \details For higher order tensors this returns the flattened value
 *
 * \param v Input index tensor
 * \return Index values
 */
std::vector<Eigen::DenseIndex> as_vector(const IndexTensor& v);

}

#endif
