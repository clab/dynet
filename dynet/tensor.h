/**
 * \file tensor.h
 * \defgroup tensor tensor
 *
 */

#ifndef DYNET_EIGEN_TENSOR_H
#define DYNET_EIGEN_TENSOR_H

#include <initializer_list>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "dynet/dim.h"
#include "dynet/except.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/devices.h"
#include "dynet/io-macros.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "dynet/cuda.h"
#endif

// Following line is commented out because it causes errors with large nets (Antonis)
//#define EIGEN_NO_MALLOC

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include <unsupported/Eigen/CXX11/Tensor>

namespace dynet {

#define EIGEN_BACKEND 1

/**
 * \ingroup tensor
 * \typedef Represents a scalar
 */
typedef float real;

/**
 * \ingroup tensor
 * \brief Represents a tensor of any order
 * \details This provides a bridge between classic C++ types and Eigen tensors.
 *
 */
struct Tensor {
  /**
   * \brief Create an empty tensor
   */
  Tensor() : d(Dim()), v(nullptr), device(nullptr), mem_pool(DeviceMempool::NONE) { }
  /**
   * \brief Creates a tensor
   * \details [long description]
   *
   * \param d Shape of the tensor
   * \param v Pointer to the values
   * \param dev Device
   * \param mem Memory pool
   */
  Tensor(const Dim& d, float* v, Device* dev, DeviceMempool mem) : d(d), v(v), device(dev), mem_pool(mem) {}
  /**
   * \brief Get the data as an Eigen matrix
   * \return Eigen matrix
   */
  Eigen::Map<Eigen::MatrixXf> operator*() {
    if(d.batch_elems() != 1 || d.ndims() >= 3)
      DYNET_INVALID_ARG("Attempted to access Tensor with more than one batch element or more than two dimensions in matrix form: " << d);
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols());
  }
  const Eigen::Map<Eigen::MatrixXf> operator*() const {
    if(d.batch_elems() != 1 || d.ndims() >= 3)
      DYNET_INVALID_ARG("Attempted to access Tensor with more than one batch element or more than two dimensions in matrix form: " << d);
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols());
  }
  /**
   * \brief Get the data as an Eigen vector
   * \details This returns the full tensor contents even if it has many dimensions
   * \return Flattened tensor
   */
  Eigen::Map<Eigen::VectorXf> vec() {
    return Eigen::Map<Eigen::VectorXf>(v, d.size());
  }
  const Eigen::Map<Eigen::VectorXf> vec() const {
    return Eigen::Map<Eigen::VectorXf>(v, d.size());
  }
  /**
   * \brief Get the data as an order 1 Eigen tensor
   * \details this returns the full tensor contents as a one dimensional Eigen tensor which can be used for on-device processing where dimensions aren't important
   * \return Eigen order 1 tensor
   */
  Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec() {
    return Eigen::TensorMap<Eigen::Tensor<float, 1>>(v, d.size());
  }
  const Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec() const {
    return Eigen::TensorMap<Eigen::Tensor<float, 1>>(v, d.size());
  }
  /**
   * \brief Get the data as an order 2 tensor including batch size
   * \details this returns the full tensor contents as a two dimensional Eigen tensor where the first dimension is a flattened representation of each batch and the second dimension is the batches
   * \return batch size x elements per batch matrix
   */
  Eigen::TensorMap<Eigen::Tensor<float, 2>> tbvec() {
    return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, d.batch_size(), d.batch_elems());
  }
  const Eigen::TensorMap<Eigen::Tensor<float, 2>> tbvec() const {
    return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, d.batch_size(), d.batch_elems());
  }
  // Get view as an Eigen Tensor (see specializations below-- this is to work Eigen's and DYNETs compile-type vs. run-time differences)
  /**
   * \brief Get view as a Tensor
   * \tparam Order Tensor order. Order 0 through 4 are already implemented for you
   * \return Eigen Tensor of the given order
   */
  template <int Order> Eigen::TensorMap<Eigen::Tensor<float, Order>> t();
  template <int Order> const Eigen::TensorMap<Eigen::Tensor<float, Order>> t() const;

  /**
   * \brief Get view as an Eigen Tensor where the final dimension is the various batches
   * \tparam Order Tensor order. Order 0 through 4 are already implemented for you
   * \return Eigen Tensor of the given order + 1
   */
  template <int Order> Eigen::TensorMap < Eigen::Tensor < float, Order + 1 >> tb();
  template <int Order> const Eigen::TensorMap < Eigen::Tensor < float, Order + 1 >> tb() const;

  /**
   * \brief Get the pointer for a particular batch
   * \details Automatically broadcasting if the size is zero
   *
   * \param bid Batch id requested
   * \return Pointer to the memory where the batch values are located
   */
  float* batch_ptr(unsigned bid) {
    DYNET_ASSERT(d.bd == 1 || bid < d.bd, "Batch index out of bounds in batch_ptr: index=" << bid << ", dim=" << d);
    return v + (bid % d.bd) * d.batch_size();
  }
  const float* batch_ptr(unsigned bid) const {
    DYNET_ASSERT(d.bd == 1 || bid < d.bd, "Batch index out of bounds in batch_ptr: index=" << bid << ", dim=" << d);
    return v + (bid % d.bd) * d.batch_size();
  }
  /**
  * \brief Get the matrix for a particular batch
  * \details Automatically broadcasting if the size is zero.
  *
  * \param bid Batch id requested
  * \return Matrix at batch id `bid` (of shape `d.rows()` x `d.cols()`)
  */
  Eigen::Map<Eigen::MatrixXf> batch_matrix(unsigned bid) {
    return Eigen::Map<Eigen::MatrixXf>(v + (bid % d.bd) * d.batch_size(), d.rows(), d.cols());
  }
  const Eigen::Map<Eigen::MatrixXf> batch_matrix(unsigned bid) const {
    return Eigen::Map<Eigen::MatrixXf>(v + (bid % d.bd) * d.batch_size(), d.rows(), d.cols());
  }
  /**
   * \brief Get the data as a matrix, where each "row" is the concatenation of rows and columns, and each "column" is batches
   * \return matrix of shape `d.rows() * d.cols()` x `d.batch_elems()`
   */
  Eigen::Map<Eigen::MatrixXf> rowcol_matrix() {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows() * d.cols(), d.batch_elems());
  }
  const Eigen::Map<Eigen::MatrixXf> rowcol_matrix() const {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows() * d.cols(), d.batch_elems());
  }

  /**
   * \brief Get the data as a matrix, where each "row" is the concatenation of rows, and each "column" is the concatenation of columns and batches
   * \return matrix of shape `d.rows() * d.cols()` x `d.batch_elems()`
   */
  Eigen::Map<Eigen::MatrixXf> colbatch_matrix() {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols() * d.batch_elems());
  }
  const Eigen::Map<Eigen::MatrixXf> colbatch_matrix() const {
    return Eigen::Map<Eigen::MatrixXf>(v, d.rows(), d.cols() * d.batch_elems());
  }
  /**
   * \brief Check for NaNs and infinite values
   * \details This is very slow: use sparingly (it's linear in the number of elements). This raises a `std::runtime_error` exception if the Tensor is on GPU because it's not implemented yet
   * \return Whether the tensor contains any invalid value
   */
  inline bool is_valid() const {
#if HAVE_CUDA
    // TODO : replace this with a custom exception
    if (device->type == DeviceType::GPU) {
      throw std::runtime_error("is_valid() not implemented on GPU");
    } else {
#endif
      const size_t s = d.size();
      for (unsigned i = 0; i < s; ++i)
        if (std::isnan(v[i]) || std::isinf(v[i])) return false;
      return true;
#if HAVE_CUDA
    }
#endif
  }

  /**
   * \brief Get a Tensor object representing a single batch.
   * \details If this tensor only has a single batch, then broadcast. Otherwise, check to make sure that the requested batch is smaller than the number of batches.
   *
   * TODO: This is a bit wasteful, as it re-calculates `bs.batch_size()` every time.
   *
   * \param b Batch id
   * \return Sub tensor at batch `b`
   */
  Tensor batch_elem(unsigned b) const {
    if (d.batch_elems() == 1) {
      return *this;
    } else {
      if (b >= d.batch_elems()) {
        std::stringstream ss;
        ss << "Requested batch id " << b << " is greater than the number of batch " << d.batch_elems();
        throw std::runtime_error(ss.str());
      }
      const unsigned bsize = d.batch_size();
      Dim new_d(d); new_d.bd = 1;
      Tensor ret(new_d, v + bsize * b, device, mem_pool);
      // std::cerr << "Getting tensor for batch " << (b % d.batch_elems()) << " bsize: " << bsize << ", ptr=" << (long)ret.v << std::endl;
      return ret;
    }
  }

  // get tensors for all batches
  /**
   * \brief Get tensors for all batches
   * \return List of the tensors in each batch
   */
  std::vector<Tensor> batch_elems() const {
    if (d.batch_elems() == 1) {
      return std::vector<Tensor>(1, *this);
    } else {
      std::vector<Tensor> bs(d.batch_elems());
      unsigned bsize = d.batch_size();
      Dim new_d = d; new_d.bd = 1;
      for (unsigned b = 0; b < d.batch_elems(); ++b)
        bs[b] = Tensor(new_d, v + bsize * b, device, mem_pool);
      return bs;
    }
  }

  Dim d;  /**< Shape of tensor */
  float* v;  /**< Pointer to memory */
  std::vector<Tensor> bs;
  Device* device;
  DeviceMempool mem_pool;

private:
  DYNET_SERIALIZE_SPLIT_DECLARE()
};

template<> inline Eigen::TensorMap<Eigen::Tensor<float, 0>> Tensor::t<0>() {
  DYNET_ASSERT(d.batch_elems() == 1 && d.size() == 1, "Illegal access of tensor in function t<0>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 0>>(v);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 0>> Tensor::t<0>() const {
  DYNET_ASSERT(d.batch_elems() == 1 && d.size() == 1, "Illegal access of tensor in function t<0>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 0>>(v);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 1>> Tensor::t<1>() {
  DYNET_ASSERT(d.batch_elems() == 1 && (d.ndims() == 1 || d.size() == d.rows()), "Illegal access of tensor in function t<1>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(v, (int)d[0]);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 1>> Tensor::t<1>() const {
  DYNET_ASSERT(d.batch_elems() == 1 && (d.ndims() == 1 || d.size() == d.rows()), "Illegal access of tensor in function t<1>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(v, (int)d[0]);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 2>> Tensor::t<2>() {
  DYNET_ASSERT(d.batch_elems() == 1 && d.ndims() <= 2, "Illegal access of tensor in function t<2>(): dim=" << d);
  if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, (int)d[0], (int)d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, (int)d[0], (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 2>> Tensor::t<2>() const {
  DYNET_ASSERT(d.batch_elems() == 1 && d.ndims() <= 2, "Illegal access of tensor in function t<2>(): dim=" << d);
  if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, (int)d[0], (int)d[1]);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, (int)d[0], (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 3>> Tensor::t<3>() {
  DYNET_ASSERT(d.batch_elems() == 1 && d.ndims() <= 3, "Illegal access of tensor in function t<3>(): dim=" << d);
  if (d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)d[1], (int)d[2]);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 3>> Tensor::t<3>() const {
  DYNET_ASSERT(d.batch_elems() == 1 && d.ndims() <= 3, "Illegal access of tensor in function t<3>(): dim=" << d);
  if (d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)d[1], (int)d[2]);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)d[1], (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)1, (int)1);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 4>> Tensor::t<4>() {
  DYNET_ASSERT(d.batch_elems() == 1 && d.ndims() <= 4, "Illegal access of tensor in function t<4>(): dim=" << d);
  if (d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3]);
  else if (d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)1, (int)1, (int)1);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 4>> Tensor::t<4>() const {
  DYNET_ASSERT(d.batch_elems() == 1 && d.ndims() <= 4, "Illegal access of tensor in function t<4>(): dim=" << d);
  if (d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3]);
  else if (d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)1, (int)1);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)1, (int)1, (int)1);
}
// ...

template<> inline Eigen::TensorMap<Eigen::Tensor<float, 1>> Tensor::tb<0>() {
  DYNET_ASSERT(d.batch_size() == 1, "Illegal access of tensor in function tb<0>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(v, (int)d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 1>> Tensor::tb<0>() const {
  DYNET_ASSERT(d.batch_size() == 1, "Illegal access of tensor in function tb<0>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 1>>(v, (int)d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 2>> Tensor::tb<1>() {
  DYNET_ASSERT(d.ndims() == 1 || d.batch_size() == d.rows(), "Illegal access of tensor in function tb<1>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, (int)d[0], (int)d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 2>> Tensor::tb<1>() const {
  DYNET_ASSERT(d.ndims() == 1 || d.batch_size() == d.rows(), "Illegal access of tensor in function tb<1>(): dim=" << d);
  return Eigen::TensorMap<Eigen::Tensor<float, 2>>(v, (int)d[0], (int)d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 3>> Tensor::tb<2>() {
  DYNET_ASSERT(d.ndims() <= 2, "Illegal access of tensor in function tb<2>(): dim=" << d);
  if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)d[1], (int)d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)1, (int)d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 3>> Tensor::tb<2>() const {
  DYNET_ASSERT(d.ndims() <= 2, "Illegal access of tensor in function tb<2>(): dim=" << d);
  if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)d[1], (int)d.bd);
  else               return Eigen::TensorMap<Eigen::Tensor<float, 3>>(v, (int)d[0], (int)1, (int)d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 4>> Tensor::tb<3>() {
  DYNET_ASSERT(d.ndims() <= 3, "Illegal access of tensor in function tb<3>(): dim=" << d);
  if (d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d.bd);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)1, (int)d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)1, (int)1, (int)d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 4>> Tensor::tb<3>() const {
  DYNET_ASSERT(d.ndims() <= 3, "Illegal access of tensor in function tb<3>(): dim=" << d);
  if (d.ndims() == 3)      return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d.bd);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)d[1], (int)1, (int)d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 4>>(v, (int)d[0], (int)1, (int)1, (int)d.bd);
}
template<> inline Eigen::TensorMap<Eigen::Tensor<float, 5>> Tensor::tb<4>() {
  DYNET_ASSERT(d.ndims() <= 4, "Illegal access of tensor in function tb<4>(): dim=" << d);
  if (d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3], (int)d.bd);
  else if (d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1, (int)d.bd);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)d[1], (int)1, (int)1, (int)d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)1, (int)1, (int)1, (int)d.bd);
}
template<> inline const Eigen::TensorMap<Eigen::Tensor<float, 5>> Tensor::tb<4>() const {
  DYNET_ASSERT(d.ndims() <= 4, "Illegal access of tensor in function tb<4>(): dim=" << d);
  if (d.ndims() == 4)      return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)d[3], (int)d.bd);
  else if (d.ndims() == 3) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)d[1], (int)d[2], (int)1, (int)d.bd);
  else if (d.ndims() == 2) return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)d[1], (int)1, (int)1, (int)d.bd);
  else                    return Eigen::TensorMap<Eigen::Tensor<float, 5>>(v, (int)d[0], (int)1, (int)1, (int)1, (int)d.bd);
}
// ...

/**
 * \ingroup tensor
 * \brief You can use `cout<<tensor;` for debugging or saving
 *
 * \param os output stream
 * \param t Tensor
 */
std::ostream& operator<<(std::ostream& os, const Tensor& t);
/**
 * \ingroup tensor
 * \brief Get a scalar value from an order 0 tensor
 * \details Throws an `runtime_error` exception if the tensor has more than one element.
 *
 * TODO : Change for custom invalid dimension exception maybe?
 *
 * \param t Input tensor
 * \return Scalar value
 */
real as_scalar(const Tensor& t);
/**
 * \ingroup tensor
 * \brief Get the array of values in the tensor
 * \details For higher order tensors this returns the flattened value
 *
 * \param v Input tensor
 * \return Values
 */
std::vector<real> as_vector(const Tensor& v);

/**
 * \ingroup tensor
 * \brief Provides tools for creating, accessing, copying and modifying tensors (in-place)
 *
 */
struct TensorTools {
  /**
   * \brief Fills the tensor with a constant value
   *
   * \param d Tensor to modify
   * \param c Target value
   */
  static void Constant(Tensor& d, float c);
  /**
   * \brief Fills a tensor with zeros
   *
   * \param d Input tensor
   */
  static void Zero(Tensor& d);
  /**
   * \brief Set the (order 2) tensor as the identity matrix
   * \details this throws a runtime_error exception if the tensor isn't a square matrix
   *
   * \param val Input tensor
   */
  static void Identity(Tensor& val);
  //
  /**
   * \brief Fill the tensor with bernoulli random variables and scale them by scale
   *
   * \param val Input tensor
   * \param p Parameter of the bernoulli distribution
   * \param scale Scale of the random variables
   */
  static void RandomizeBernoulli(Tensor& val, real p, real scale = 1.0f);
  /**
   * \brief Fill the tensor with gaussian random variables
   *
   * \param val Input tensor
   * \param mean Mean
   * \param stddev Standard deviation
   */
  static void RandomizeNormal(Tensor& val, real mean = 0.0f, real stddev = 1.0f);
  /**
   * \brief Fill the tensor with uniform random variables
   *
   * \param val Input tensor
   * \param left Left bound of the interval
   * \param right Right bound of the interval
   */
  static void RandomizeUniform(Tensor& val, real left = 0.0f, real right = 0.0f);
  /**
   * \brief Takes a square matrix tensor and sets it as a random orthonormal matrix
   * \details More specifically this samples a random matrix with RandomizeUniform and then performs SVD and returns the left orthonormal matrix in the decomposition, scaled by `scale`
   *
   * \param val Input tensor
   * \param scale Value to which the resulting orthonormal matrix will be scaled
   */
  static void RandomizeOrthonormal(Tensor& val, real scale = 1.0f);
  /**
   * \brief Access element of the tensor by index in the values array
   * \details AccessElement and SetElement are very, very slow (potentially) - use appropriately
   *
   * \param v Tensor
   * \param index Index in the memory
   *
   * \return `v.v[index]`
   */
  static float AccessElement(const Tensor& v, int index);
  /**
   * \brief Access element of the tensor by indices in the various dimension
   * \details This only works for matrix shaped tensors (+ batch dimension). AccessElement and SetElement are very, very slow (potentially) - use appropriately
   *
   * \param v Tensor
   * \param index Indices in the tensor
   *
   * \return `(*v)(index[0], index[1])`
   */
  static float AccessElement(const Tensor& v, const Dim& index);
  /**
   * \brief Set element of the tensor by index in the values array
   * \details AccessElement and SetElement are very, very slow (potentially) - use appropriately
   *
   * \param v Tensor
   * \param index Index in the memory
   * \param value Desired value
   */
  static void SetElement(const Tensor& v, int index, float value);
  /**
   * \brief Copy element from one tensor to another (by index in the values array)
   *
   * \param l Source tensor
   * \param lindex Source index
   * \param r Target tensor
   * \param rindex Target index
   */
  static void CopyElement(const Tensor& l, int lindex, Tensor& r, int rindex);

  /**
   * \brief Set the elements of a tensor with an array of values
   * \details (This uses memcpy so be careful)
   *
   * \param v Input Tensor
   * \param vec Values
   */
  static void SetElements(const Tensor& v, const std::vector<float>& vec);
  /**
   * \brief Copy one tensor into another
   *
   * \param v Target tensor
   * \param v_src Source tensor
   */
  static void CopyElements(const Tensor& v, const Tensor& v_src);
};

/**
 * \ingroup tensor
 * \brief This is a helper function to sample uniformly in \f$[0,1]\f$
 * \return \f$x\sim\mathcal U([0,1])\f$
 */
real rand01();
/**
 * \ingroup tensor
 * \brief This is a helper function to sample uniformly in \f$\{0,\dots,n-1\}\f$
 *
 * \param n Upper bound (excluded)
 * \return \f$x\sim\mathcal U(\{0,\dots,n-1\})\f$
 */
int rand0n(int n);
/**
 * \ingroup tensor
 * \brief This is a helper function to sample from a normalized gaussian distribution
 *
 * \return \f$x\sim\mathcal N(0,1)\f$
 */
real rand_normal();

} // namespace dynet

DYNET_VERSION_DEFINE(dynet::Tensor, 1)
#endif
