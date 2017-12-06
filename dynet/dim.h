/**
 * \defgroup dim dim
 * \ingroup dim
 * \file dim.h
 * \brief Dynet's way of implementing minibatching
 */

#ifndef DYNET_DIM_H
#define DYNET_DIM_H

#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <iosfwd>
#include <cstring>
#include <vector>

#include "dynet/except.h"

/**
 * \ingroup dim
 * Maximum number of dimensions supported by dynet : 7
 */
#define DYNET_MAX_TENSOR_DIM 7

namespace dynet {

/**
 * \ingroup dim
 * \struct Dim
 * \brief The Dim struct stores information about the dimensionality of expressions.
 * \details Batch dimension is treated separately from standard dimension.
 */
struct Dim {
  /**
   * \brief Default constructor
   */
  Dim() : nd(0), bd(1) {}
  // explicit Dim(unsigned int m) : nd(1), bd(1) { d[0] = m; }
  // TODO: The constructors for dimensions w/ and w/o batches is not intuitive.
  //       can this be fixed in some way?
  // Dim(unsigned int m, unsigned int n) : nd(2), bd(1) { d[0] = m; d[1] = n; }
  /**
   * \brief Initialize from a list of dimensions
   * \details The batch dimension is 1 in this case (non-batched expression)
   *
   * \param x List of dimensions
   */
  Dim(std::initializer_list<unsigned int> x) : nd(0), bd(1) {
    DYNET_ARG_CHECK(
        x.size() <= DYNET_MAX_TENSOR_DIM,
        "Out of bounds exception in Dim::Dim() with initializer_list of size "
        << x.size());
    for (auto v : x) d[nd++] = v;
  }
  /**
   * \brief Initialize from a list of dimensions and a batch size
   *
   * \param x  List of dimensions
   * \param b Batch size
   */
  Dim(std::initializer_list<unsigned int> x, unsigned int b) : nd(0), bd(b) {
    DYNET_ARG_CHECK(
        x.size() <= DYNET_MAX_TENSOR_DIM,
        "Out of bounds exception in Dim::Dim() with initializer_list of size "
        << x.size());
    for (auto v : x) d[nd++] = v;
  }
  /**
   * \brief Initialize from a vector of dimensions
   * \details The batch dimension is 1 in this case (non-batched expression)
   *
   * \param x Array of dimensions
   */
  Dim(const std::vector<long> & x) : nd(0), bd(1) {
    DYNET_ARG_CHECK(
        x.size() <= DYNET_MAX_TENSOR_DIM,
        "Out of bounds exception in Dim::Dim() with vector of size "
        << x.size());
    for (auto v : x) d[nd++] = static_cast<unsigned int>(v);
  }
  /**
     * \brief Initialize from a vector of dimensions and a batch size
     *
     * \param x Vector of dimensions
     * \param b Batch size
     */
  Dim(const std::vector<long> & x, unsigned int b) : nd(0), bd(b) {
    DYNET_ARG_CHECK(
        x.size() <= DYNET_MAX_TENSOR_DIM,
        "Out of bounds exception in Dim::Dim() with vector of size "
        << x.size());
    for (auto v : x) d[nd++] = static_cast<unsigned int>(v);
  }
  /**
   * \brief Total size of a batch
   * \return Batch size * size of a batch
   */
  inline unsigned int size() const {
    return batch_size() * bd;
  }
  /**
   * \brief Size of a batch (product of all dimensions)
   * \return Size of a batch
   */
  inline unsigned int batch_size() const {
    unsigned int p = 1;
    for (unsigned int i = 0; i < nd; ++i) p *= d[i];
    return p;
  }
  /**
   * \brief Sum of all dimensions within a batch
   * \return Sum of the dimensions within a batch
   */
  inline unsigned int sum_dims() const {
    unsigned int p = 0;
    for (unsigned int i = 0; i < nd; ++i) p += d[i];
    return p;
  }
  /**
   * \brief remove trailing dimensions of 1
   * \details iterate all the dimensions of Dim, stop at last dimension of 1
   * \return truncated dimension
   */
  inline Dim truncate() const {
    Dim r = *this;
    unsigned int m = nd;
    while (m > 1 && size(m-1) == 1) --m;
    r.resize(m);
    return r;
  }
  /**
   * \brief Set the batch dimension to 1
   * \return 1-batch version of this instance
   */
  inline Dim single_batch() const {
    Dim r = *this;
    r.bd = 1;
    return r;
  }
  /**
   * \brief Change the number of dimensions
   *
   * \param int New number of dimensions
   */
  inline void resize(unsigned int i) {
    while(nd < i)
      d[nd++] = 1;
    nd = i;
  }
  /**
   * \brief Get number of dimensions
   * \return Number of dimensions
   */
  inline unsigned int ndims() const { return nd; }
  /**
   * \brief Size of the first dimension
   * \return Size of the first dimension
   */
  inline unsigned int rows() const { return d[0]; }
  /**
   * \brief Number of non-one dimensions
   * \return Number of non-one dimensions
   */
  inline unsigned int num_nonone_dims() const {
    int ret = 0;
    for(size_t i = 0; i < nd; ++i)
      if(d[i] != 1)
        ++ret;
    return ret;
  }
  /**
   * \brief Size of the second dimension (or 1 if only one dimension)
   * \return Size of the second dimension (or 1 if only one dimension)
   */
  inline unsigned int cols() const { return nd > 1 ? d[1] : 1; }
  /**
   * \brief Batch dimension
   * \return Batch dimension
   */
  inline unsigned int batch_elems() const { return bd; }
  /**
   * \brief Set specific dimension
   * \details Set the value of a specific dimension to an arbitrary value
   *
   * \param i Dimension index
   * \param s Dimension size
   */
  inline void set(unsigned int i, unsigned int s) {
    DYNET_ARG_CHECK(i < nd || s == 1, "Out of bounds exception in Dim::set(" << i << "," << s << ") for node of size " << nd);
    DYNET_ARG_CHECK(s != 0, "Attempt to set dimension size to zero in Dim::set(" << i << "," << s << ") for node of size " << nd);
    d[i] = s;
  }
  /**
   * \brief Access a specific dimension as you would access an array element
   *
   * \param i Dimension index
   * \return Size of dimension i
   */
  inline unsigned int operator[](unsigned int i) const { return i < nd ? d[i] : 1; }
  /**
   * \brief Size of dimension i
   *
   * \param i Dimension index
   * \return Size of dimension i
   */
  inline unsigned int size(unsigned int i) const { return (*this)[i]; }
  /**
   * \brief Remove one of the dimensions
   * \param i index of the dimension to be removed
   */
  inline void delete_dim(unsigned int i) {
    DYNET_ARG_CHECK(i < nd, "Out of bounds exception in Dim::delete_dim(" << i << ") for node of size " << nd );
    if(i == nd-1){
      if(nd == 1){
        d[0] = 1;
      }
      else{
        --nd;
      }
    }
    else{
      for(; i + 1 < nd; ++i){
        d[i] = d[i + 1];
      }
      --nd;
    }
  }
  /**
   * \brief Remove multi-dimensions
   * \param dims dimensions to be removed
   * \param reduce_batch reduce the batch dimension or not
   */
  inline void delete_dims(std::vector<unsigned int> dims, bool reduce_batch){
    std::vector<bool> deleted_dims(nd, false);

    for(unsigned int i = 0; i < dims.size(); i++) {
      DYNET_ARG_CHECK(dims[i] < nd, "Out of bounds exception in Dim::delete_dims");
      deleted_dims[dims[i]] = true;
    }

    if(dims.size() == nd) {
        nd = 1;
        d[0] = 1;
    } else {
      int flag = 0;
      for(unsigned int i = 0; i < nd; i++) {
        if(!deleted_dims[i])
          d[flag++] = d[i];
      }
      nd = flag;
    }

    if(reduce_batch)
      bd = 1;
  }
  /**
   * \brief Insert a dimension to the end
   * \param n the size of the new dimension
   */
  inline void add_dim(unsigned int n) {
    DYNET_ARG_CHECK(nd + 1 <= DYNET_MAX_TENSOR_DIM, "Out of bounds exception in Dim::add_dim");
    d[nd] = n;
    nd++;
  }
  /**
   * \brief Insert a dimension
   * \param i the index to insert the new dimension
   * \param n the size of the new dimension
   */
  inline void insert_dim(unsigned int i, unsigned int n) {
    DYNET_ARG_CHECK(nd + 1 <= DYNET_MAX_TENSOR_DIM, "Out of bounds exception in Dim::add_dim");
    DYNET_ARG_CHECK(i <= nd, "Out of bounds exception in Dim::insert_dim(" << i << ") for node of size " << nd);
    if (nd == 0) {
      d[0] = n;
    } else {
      for (int k = nd; k > (int)i; --k) {
        d[k] = d[k-1];
      }
    }
    d[i] = n;
    ++nd;
  }
  /**
  * \brief Transpose a vector or a matrix
  * \details This raises an invalid_argument exception on tensors with more than 2 dimensions
  * \return The transposed Dim structure
  */
  inline Dim transpose() const {
    if (nd == 1) { return Dim({1, d[0]}, bd); }
    else {
      DYNET_ARG_CHECK(nd == 2, "Cannot transpose Dim object with more than 2 dimensions, but got " << nd);
      return Dim({d[1], d[0]}, bd);
    }
  }
  /**
  * \brief Print the unbatched profile as a string
  **/
  void print_profile(std::ostream & out) const;

  unsigned int d[DYNET_MAX_TENSOR_DIM]; /**< Array of dimension */
  unsigned int nd; /**< Number of dimensions */
  unsigned int bd; /**< Batch dimension */
};

/**
 * \brief Check for equality between two Dim
 * \details Two Dim struct are considered equal if their dimensions and batch size are equal
 * 
 * \param a First Dim
 * \param b Second Dim
 * 
 * \return a==b
 */
inline bool operator==(const Dim& a, const Dim& b) {
  if (a.nd != b.nd || a.bd != b.bd) return false;
  return std::memcmp(a.d, b.d, a.nd * sizeof(unsigned int)) == 0;
}

/**
 * \brief Check for inequality of two Dim structs
 * \details See equality
 * 
 * \param a First Dim
 * \param b Second Dim
 * 
 * \return a!=b
 */
inline bool operator!=(const Dim& a, const Dim& b) { return !(a == b); }

/**
 * \brief Print Dim to output stream
 * 
 * \param os Output stream
 * \param d Dim
 */
std::ostream& operator<<(std::ostream& os, const Dim& d);
/**
 * \brief Print vector of Dims to output stream
 * 
 * \param os Output stream
 * \param ds vector of Dims
 */
std::ostream& operator<<(std::ostream& os, const std::vector<Dim>& ds);

std::istream& operator>>(std::istream& os, Dim& d);

} // namespace dynet

#endif
