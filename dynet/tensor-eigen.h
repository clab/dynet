#ifndef DYNET_TENSOR_EIGEN_H
#define DYNET_TENSOR_EIGEN_H

// This file includes all of the DyNet tensor functions that require
// Eigen to be importet.d. It should be included sparingly to prevent
// unnecessary compile time.

#include "dynet/tensor.h"

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include <unsupported/Eigen/CXX11/Tensor>

namespace dynet {

/**
 * \brief Get the data as an Eigen matrix
 * \return Eigen matrix
 */
inline Eigen::Map<Eigen::MatrixXf> mat(Tensor& t) {
  DYNET_ARG_CHECK((t.d.batch_elems() == 1 && t.d.ndims() < 3),
                          "Attempted to access Tensor with more than one batch element or more than two dimensions in matrix form: " << t.d);
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols());
}
inline const Eigen::Map<Eigen::MatrixXf> mat(const Tensor& t) {
  DYNET_ARG_CHECK((t.d.batch_elems() == 1 && t.d.ndims() < 3),
                          "Attempted to access Tensor with more than one batch element or more than two dimensions in matrix form: " << t.d);
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols());
}
/**
 * \brief Get the data as an Eigen vector
 * \details This returns the full tensor contents even if it has many dimensions
 * \return Flattened tensor
 */
inline Eigen::Map<Eigen::VectorXf> vec(Tensor & t) {
  return Eigen::Map<Eigen::VectorXf>(t.v, t.d.size());
}
inline const Eigen::Map<Eigen::VectorXf> vec(const Tensor & t) {
  return Eigen::Map<Eigen::VectorXf>(t.v, t.d.size());
}

/**
 * \brief Get the data as an order 1 Eigen tensor
 * \details this returns the full tensor contents as a one dimensional Eigen tensor which can be used for on-device processing where dimensions aren't important
 * \return Eigen order 1 tensor
 */
inline Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec(Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, t.d.size());
}
inline const Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec(const Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, t.d.size());
}
/**
 * \brief Get the data as an order 2 tensor including batch size
 * \details this returns the full tensor contents as a two dimensional Eigen tensor where the first dimension is a flattened representation of each batch and the second dimension is the batches
 * \return batch size x elements per batch matrix
 */
inline Eigen::TensorMap<Eigen::Tensor<float, 2>> tbvec(Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, t.d.batch_size(), t.d.batch_elems());
}
inline const Eigen::TensorMap<Eigen::Tensor<float, 2>> tbvec(const Tensor & t) {
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, t.d.batch_size(), t.d.batch_elems());
}
// Get view as an Eigen Tensor (see specializations below-- this is to work Eigen's and DyNet's compile-type vs. run-time differences)
/**
 * \brief Get view as a Tensor
 * \tparam Order Tensor order. Order 0 through 4 are already implemented for you
 * \return Eigen Tensor of the given order
 */
template <int Order> inline Eigen::TensorMap<Eigen::Tensor<float, Order>> t(Tensor & t);
template <int Order> inline const Eigen::TensorMap<Eigen::Tensor<float, Order>> t(const Tensor & t);

template<> inline Eigen::TensorMap<Eigen::Tensor<float, 0>> t<0>(Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.size() == 1, "Illegal access of tensor in function t<0>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 0>>(t.v);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 0>> t<0>(const Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.size() == 1, "Illegal access of tensor in function t<0>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 0>>(t.v);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 1>> t<1>(Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && (t.d.ndims() == 1 || t.d.size() == t.d.rows()), "Illegal access of tensor in function t<1>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, (int)t.d[0]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 1>> t<1>(const Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && (t.d.ndims() == 1 || t.d.size() == t.d.rows()), "Illegal access of tensor in function t<1>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, (int)t.d[0]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 2>> t<2>(Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 2, "Illegal access of tensor in function t<2>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, (int)t.d[0], (int)t.d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, (int)t.d[0], (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 2>> t<2>(const Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 2, "Illegal access of tensor in function t<2>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, (int)t.d[0], (int)t.d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, (int)t.d[0], (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 3>> t<3>(Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 3, "Illegal access of tensor in function t<3>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2]);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 3>> t<3>(const Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 3, "Illegal access of tensor in function t<3>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2]);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)1, (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 4>> t<4>(Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 4, "Illegal access of tensor in function t<4>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3]);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 4>> t<4>(const Tensor & t) {
  DYNET_ASSERT(t.d.batch_elems() == 1 && t.d.ndims() <= 4, "Illegal access of tensor in function t<4>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3]);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1);
}
// ...

/**
 * \brief Get view as an Eigen Tensor where the final dimension is the various batches
 * \tparam Order Tensor order. Order 0 through 4 are already implemented for you
 * \return Eigen Tensor of the given order + 1
 */
template <int Order> Eigen::TensorMap < Eigen::Tensor < float, Order + 1 >> tb(Tensor & t);
template <int Order> const Eigen::TensorMap < Eigen::Tensor < float, Order + 1 >> tb(const Tensor & t);

template<> inline Eigen::TensorMap<Eigen::Tensor<float, 1>> tb<0>(Tensor & t) {
  DYNET_ASSERT(t.d.batch_size() == 1, "Illegal access of tensor in function tb<0>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 1>> tb<0>(const Tensor & t) {
  DYNET_ASSERT(t.d.batch_size() == 1, "Illegal access of tensor in function tb<0>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(t.v, (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 2>> tb<1>(Tensor & t) {
  DYNET_ASSERT(t.d.ndims() == 1 || t.d.batch_size() == t.d.rows(), "Illegal access of tensor in function tb<1>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, (int)t.d[0], (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 2>> tb<1>(const Tensor & t) {
  DYNET_ASSERT(t.d.ndims() == 1 || t.d.batch_size() == t.d.rows(), "Illegal access of tensor in function tb<1>(Tensor & t): dim=" << t.d);
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(t.v, (int)t.d[0], (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 3>> tb<2>(Tensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 2, "Illegal access of tensor in function tb<2>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)1, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 3>> tb<2>(const Tensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 2, "Illegal access of tensor in function tb<2>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 3>>(t.v, (int)t.d[0], (int)1, (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 4>> tb<3>(Tensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 3, "Illegal access of tensor in function tb<3>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 4>> tb<3>(const Tensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 3, "Illegal access of tensor in function tb<3>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(t.v, (int)t.d[0], (int)1, (int)1, (int)t.d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 5>> tb<4>(Tensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 4, "Illegal access of tensor in function tb<4>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3], (int)t.d.bd);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1, (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1, (int)t.d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 5>> tb<4>(const Tensor & t) {
  DYNET_ASSERT(t.d.ndims() <= 4, "Illegal access of tensor in function tb<4>(Tensor & t): dim=" << t.d);
  if (t.d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)t.d[3], (int)t.d.bd);
  else if (t.d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)t.d[2], (int)1, (int)t.d.bd);
  else if (t.d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)t.d[1], (int)1, (int)1, (int)t.d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 5>>(t.v, (int)t.d[0], (int)1, (int)1, (int)1, (int)t.d.bd);
}
// ...


/**
* \brief Get the matrix for a particular batch
* \details Automatically broadcasting if the size is zero.
*
* \param bid Batch id requested
* \return Matrix at batch id `bid` (of shape `t.d.rows()` x `t.d.cols()`)
*/
inline Eigen::Map<Eigen::MatrixXf> batch_matrix(Tensor & t, unsigned bid) {
  return Eigen::Map<Eigen::MatrixXf>(t.v + (bid % t.d.bd) * t.d.batch_size(), t.d.rows(), t.d.cols());
}
inline const Eigen::Map<Eigen::MatrixXf> batch_matrix(const Tensor & t, unsigned bid) {
  return Eigen::Map<Eigen::MatrixXf>(t.v + (bid % t.d.bd) * t.d.batch_size(), t.d.rows(), t.d.cols());
}
/**
 * \brief Get the data as a matrix, where each "row" is the concatenation of rows and columns, and each "column" is batches
 * \return matrix of shape `t.d.rows() * t.d.cols()` x `t.d.batch_elems()`
 */
inline Eigen::Map<Eigen::MatrixXf> rowcol_matrix(Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows() * t.d.cols(), t.d.batch_elems());
}
inline const Eigen::Map<Eigen::MatrixXf> rowcol_matrix(const Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows() * t.d.cols(), t.d.batch_elems());
}

/**
 * \brief Get the data as a matrix, where each "row" is the concatenation of rows, and each "column" is the concatenation of columns and batches
 * \return matrix of shape `t.d.rows() * t.d.cols()` x `t.d.batch_elems()`
 */
inline Eigen::Map<Eigen::MatrixXf> colbatch_matrix(Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols() * t.d.batch_elems());
}
inline const Eigen::Map<Eigen::MatrixXf> colbatch_matrix(const Tensor & t) {
  return Eigen::Map<Eigen::MatrixXf>(t.v, t.d.rows(), t.d.cols() * t.d.batch_elems());
}


}

#endif
