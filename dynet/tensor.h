/**
 * \file tensor.h
 * \defgroup tensor tensor
 *
 */

#ifndef DYNET_TENSOR_H
#define DYNET_TENSOR_H

#include <initializer_list>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "dynet/dim.h"
#include "dynet/except.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/device-structs.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "dynet/cuda.h"
#endif

namespace dynet {

struct IndexTensor;

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
   * \brief Check for NaNs and infinite values
   * \details This is very slow: use sparingly (it's linear in the number of elements). This raises a `std::runtime_error` exception if the Tensor is on GPU because it's not implemented yet
   * \return Whether the tensor contains any invalid value
   */
  bool is_valid() const;

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
  Device* device;
  DeviceMempool mem_pool;
};

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
 * \brief Get the array of values in the scaled tensor
 * \details For higher order tensors this returns the flattened value
 *
 * \param v Input tensor
 * \param a Scale factor
 * \return Values
 */
std::vector<real> as_scale_vector(const Tensor& v, float a);

/**
 * \ingroup tensor
 * \brief Provides tools for creating, accessing, copying and modifying tensors (in-place)
 *
 */
struct TensorTools {
  /**
   * \brief Clip the values in the tensor to a fixed range
   *
   * \param d Tensor to modify
   * \param left Target minimum value
   * \param right Target maximum value 
   */
  static void clip(Tensor& d, float left, float right);
  /**
   * \brief Do an elementwise linear transform of values a*x + b
   *
   * \param x Tensor to modify
   * \param a The value to multiply by
   * \param b The value to add
   */
  static void scale(Tensor& x, float left, float right);
  /**
   * \brief Take a tensor of Uniform(0,1) sampled variables and turn them
   *        into Bernoulli(p) variables
   *
   * \param x Tensor to modify
   * \param p The bernoulli probability
   */
  static void uniform_to_bernoulli(Tensor& x, float p);
  /**
   * \brief Fills the tensor with a constant value
   *
   * \param d Tensor to modify
   * \param c Target value
   */
  static void constant(Tensor& d, float c);
  /**
   * \brief Fills a tensor with zeros
   *
   * \param d Input tensor
   */
  static void zero(Tensor& d);
  /**
   * \brief Set the (order 2) tensor as the identity matrix
   * \details this throws a runtime_error exception if the tensor isn't a square matrix
   *
   * \param val Input tensor
   */
  static void identity(Tensor& val);
  //
  /**
   * \brief Fill the tensor with bernoulli random variables and scale them by scale
   *
   * \param val Input tensor
   * \param p Parameter of the bernoulli distribution
   * \param scale Scale of the random variables
   */
  static void randomize_bernoulli(Tensor& val, real p, real scale = 1.0f);
  /**
   * \brief Fill the tensor with gaussian random variables
   *
   * \param val Input tensor
   * \param mean Mean
   * \param stddev Standard deviation
   */
  static void randomize_normal(Tensor& val, real mean = 0.0f, real stddev = 1.0f);
  /**
   * \brief Fill the tensor with uniform random variables
   *
   * \param val Input tensor
   * \param left Left bound of the interval
   * \param right Right bound of the interval
   */
  static void randomize_uniform(Tensor& val, real left = 0.0f, real right = 1.0f);
  /**
   * \brief Takes a square matrix tensor and sets it as a random orthonormal matrix
   * \details More specifically this samples a random matrix with RandomizeUniform and then performs SVD and returns the left orthonormal matrix in the decomposition, scaled by `scale`
   *
   * \param val Input tensor
   * \param scale Value to which the resulting orthonormal matrix will be scaled
   */
  static void randomize_orthonormal(Tensor& val, real scale = 1.0f);
  /**
   * \brief Access element of the tensor by index in the values array
   * \details AccessElement and SetElement are very, very slow (potentially) - use appropriately
   *
   * \param v Tensor
   * \param index Index in the memory
   *
   * \return `v.v[index]`
   */
  static float access_element(const Tensor& v, int index);
  /**
   * \brief Access element of the tensor by indices in the various dimension
   * \details This only works for matrix shaped tensors (+ batch dimension). AccessElement and SetElement are very, very slow (potentially) - use appropriately
   *
   * \param v Tensor
   * \param index Indices in the tensor
   *
   * \return `(*v)(index[0], index[1])`
   */
  static float access_element(const Tensor& v, const Dim& index);
  /**
   * \brief Set element of the tensor by index in the values array
   * \details AccessElement and SetElement are very, very slow (potentially) - use appropriately
   *
   * \param v Tensor
   * \param index Index in the memory
   * \param value Desired value
   */
  static void set_element(const Tensor& v, int index, float value);
  /**
   * \brief Copy element from one tensor to another (by index in the values array)
   *
   * \param l Source tensor
   * \param lindex Source index
   * \param r Target tensor
   * \param rindex Target index
   */
  static void copy_element(const Tensor& l, int lindex, Tensor& r, int rindex);

  /**
   * \brief Set the elements of a tensor with an array of values
   * \details (This uses memcpy so be careful)
   *
   * \param v Input Tensor
   * \param vec Values
   */
  static void set_elements(const Tensor& v, const std::vector<float>& vec);
  /**
   * \brief Copy one tensor into another
   *
   * \param v Target tensor
   * \param v_src Source tensor
   */
  static void copy_elements(Tensor& v, const Tensor& v_src);

  /**
   * \brief Accumulate the values of one tensor into another
   *
   * \param v Target tensor
   * \param v_src Source tensor
   */
  static void accumulate(Tensor& v, const Tensor& v_src);

  /**
  * \brief Calculate the logsumexp function over all columns of the tensor
  *
  * \param x The input tensor
  * \param m A tensor of scratch memory to hold the maximum values of each column
  * \param z The output tensor
  */
  static void logsumexp(const Tensor& x, Tensor &m, Tensor &z, unsigned d = 0);

  /**
   * \brief Calculate the index of the maximum value
   *
   * \param v A tensor where each row represents a probability distribution
   * \param dim Which dimension to take the argmax over
   * \param num The number of kmax values
   *
   * \returns A newly allocated LongTensor consisting of argmax IDs. The length of the
   *          dimension "dim" will be "num", consisting of the appropriate IDs.
   */
  static IndexTensor argmax(const Tensor& v, unsigned dim = 0, unsigned num = 1);

  /**
   * \brief Calculate samples from a log probability
   *
   * \param v A tensor where each row represents a log probability distribution
   * \param dim Which dimension to take the sample over
   * \param num The number of samples for each row
   *
   * \returns A newly allocated LongTensor consisting of argmax IDs. The length of the
   *          dimension "dim" will be "num", consisting of the appropriate IDs.
   */
  static IndexTensor categorical_sample_log_prob(const Tensor& v, unsigned dim = 0, unsigned num = 1);

  // Device functions that can be called directly if the device is already known
  template<class MyDevice>
  static void clip_dev(const MyDevice & dev, Tensor& d, float left, float right);
  template<class MyDevice>
  static void constant_dev(const MyDevice & dev, Tensor& d, float c);
  template<class MyDevice>
  static void accumulate_dev(const MyDevice & dev, Tensor& v_src, const Tensor& v);
  template<class MyDevice>
  static void scale_dev(const MyDevice & dev, Tensor& x, float a, float b);
  template<class MyDevice>
  static void uniform_to_bernoulli_dev(const MyDevice & dev, Tensor& x, float p);
  template<class MyDevice>
  static IndexTensor argmax_dev(const MyDevice & dev, const Tensor& v, unsigned dim = 0, unsigned num = 1);
  template<class MyDevice>
  static IndexTensor categorical_sample_log_prob_dev(const MyDevice & dev, const Tensor& v, unsigned dim = 0, unsigned num = 1);
  template <class MyDevice>
  static void logsumexp_dev(const MyDevice & dev, const Tensor& x, Tensor &m, Tensor &z, unsigned d = 0);

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

#endif
