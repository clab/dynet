/**
 * \file expr.h
 * \defgroup operations
 * \defgroup inputoperations
 * \defgroup arithmeticoperations
 * \defgroup lossoperations
 * \defgroup flowoperations
 * \defgroup noiseoperations
 * \defgroup convoperations
 * \defgroup tensoroperations
 * \defgroup linalgoperations
 * \brief The various operations that you can use in building a DyNet graph
 * 
 * \details TODO: Create documentation and explain expressions, etc...
 */

#ifndef DYNET_EXPR_H
#define DYNET_EXPR_H

#include "dynet/dynet.h"
#include "dynet/nodes.h"
#include "dynet/nodes-contract.h"

namespace dynet { namespace expr {
/**
 * \ingroup operations
 * \brief Expressions are the building block of a Dynet computation graph 
 * \details [long description]
 */
struct Expression {
  ComputationGraph *pg;
  VariableIndex i;

  Expression() : pg(nullptr) { }
  /**
   * \brief Base expression constructor
   * \details [long description]
   * 
   * \param pg [description]
   * \param i [description]
   */
  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i) { }
  const Tensor& value() const { return pg->get_value(i); }
};

namespace detail {
  template <typename F, typename T>
  Expression f(const T& xs) {
    ComputationGraph *pg = xs.begin()->pg;
    std::vector<VariableIndex> xis(xs.size());
    int i = 0;
    for (auto xi = xs.begin(); xi != xs.end(); ++xi) xis[i++] = xi->i;
    return Expression(pg, pg->add_function<F>(xis));
  }
}

////////////////////////////////////////////////
// Input operations                           //
////////////////////////////////////////////////

/**
 * \ingroup inputoperations
 * \brief Scalar input
 * \details Create an expression that represents the scalar value s
 * 
 * \param g Computation graph
 * \param s Real number
 * 
 * \return The new expression
 */
Expression input(ComputationGraph& g, real s);

/**
 * \ingroup inputoperations
 * \brief Modifiable scalar input
 * \details Create an expression that represents the scalar value *ps.
 *          If *ps is changed and the computation graph recalculated, the
 *          next forward pass will reflect the new value.
 * 
 * \param g Computation graph
 * \param ps Real number pointer
 * 
 * \return The new expression
 */
Expression input(ComputationGraph& g, const real *ps);

/**
 * \ingroup inputoperations
 * \brief Vector/matrix/tensor input
 * \details Create an expression that represents a vector, matrix, or tensor
 *          input. The dimensions of the input are defined by `d`. So for example
 *          > input(g,{50},data): will result in a 50-length vector
 *          > input(g,{50,30},data): will result in a 50x30 matrix
 *          and so on, for an arbitrary number of dimensions.
 *          This function can also be used to import minibatched inputs. For example,
 *          if we have 10 examples in a minibatch, each with size 50x30, then we call
 *          > input(g,Dim({50,30},10),data)
 *          The data vector "data" will contain the values used to fill the input, in
 *          column-major format. The length must add to the product of all dimensions in
 *          d.
 * 
 * \param g Computation graph
 * \param d Dimension of the input matrix
 * \param data A vector of data points
 * 
 * \return The new expression
 */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>& data);

/**
 * \ingroup inputoperations
 * \brief Updatable vector/matrix/tensor input
 * \details Similarly to input that takes a vector reference, input a vector, matrix,
 *          or tensor input. Because we pass the pointer, the data can be updated.
 * 
 * \param g Computation graph
 * \param d Dimension of the input matrix
 * \param data A pointer to an (updatable) vector of data points
 * 
 * \return The new expression
 */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);

/**
 * \ingroup inputoperations
 * \brief Sparse vector input
 * \details This operation takes input as a sparse matrix of index/value pairs. It is
 *          exactly the same as the standard input via vector reference, but sets all
 *          non-specified values to "defdata" and resets all others to the appropriate
 *          input values.
 * 
 * \param g Computation graph
 * \param d Dimension of the input matrix
 * \param ids The indexes of the data points to update
 * \param data The data points corresponding to each index
 * \param defdata The default data with which to set the unspecified data points
 * 
 * \return The new expression
 */
Expression input(ComputationGraph& g, const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);

/**
 * \ingroup inputoperations
 * \brief Load parameter
 * \details Load parameters into the computation graph.
 * 
 * \param g Computation graph
 * \param p Parameter object to load
 * 
 * \return The new expression
 */
Expression parameter(ComputationGraph& g, Parameter p);

/**
 * \ingroup inputoperations
 * \brief Load constant parameters
 * \details Load parameters into the computation graph, but prevent them from being
 *          updated when performing parameter update.
 * 
 * \param g Computation graph
 * \param p Parameter object to load
 * 
 * \return The new expression
 */
Expression const_parameter(ComputationGraph& g, Parameter p);

/**
 * \ingroup inputoperations
 * \brief Look up parameter
 * \details Look up parameters according to an index, and load them into the 
 *          computation graph.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param index Index of the parameters within p
 * 
 * \return The new expression
 */
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index);

/**
 * \ingroup inputoperations
 * \brief Look up parameters with modifiable index
 * \details Look up parameters according to the *pindex, and load them into the 
 *          computation graph. When *pindex changes, on the next computation of 
 *          forward() the values will change.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindex Pointer index of the parameters within p
 * 
 * \return The new expression
 */
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);

/**
 * \ingroup inputoperations
 * \brief Look up parameter
 * \details Look up parameters according to an index, and load them into the 
 *          computation graph. Do not perform gradient update on the parameters.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param index Index of the parameters within p
 * 
 * \return The new expression
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index);

/**
 * \ingroup inputoperations
 * \brief Constant lookup parameters with modifiable index
 * \details Look up parameters according to the *pindex, and load them into the 
 *          computation graph. When *pindex changes, on the next computation of 
 *          forward() the values will change. However, gradient updates will not be
            performend.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindex Pointer index of the parameters within p
 * 
 * \return The new expression
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);

// Batched versions of lookup and const_lookup

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details The mini-batched version of lookup. The resulting expression will be
 *          a mini-batch of parameters, where the "i"th element of the batch corresponds
 *          to the parameters at the position specified by the "i"th element of
 *          "indices"
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param indices Index of the parameters at each position in the batch
 * 
 * \return The new expression
 */
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details The mini-batched version of lookup with modifiable parameter indices.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param indices Pointer to lookup indices
 * 
 * \return The new expression
 */
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details Mini-batched lookup that will not update the parameters.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param indices Lookup indices
 * 
 * \return The new expression
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);

/**
 * \ingroup inputoperations
 * \brief Look up parameters
 * \details Mini-batched lookup that will not update the parameters, with modifiable
 *          indices.
 * 
 * \param g Computation graph
 * \param p LookupParameter object from which to load
 * \param pindices Lookup index pointers.
 * 
 * \return The new expression
 */
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);

/**
 * \ingroup inputoperations
 * \brief Create an input full of zeros
 * \details Create an input full of zeros, sized according to dimensions d.
 * 
 * \param g Computation graph
 * \param d The dimensions of the input
 * 
 * \return The new expression
 */
Expression zeroes(ComputationGraph& g, const Dim& d);

/**
 * \ingroup inputoperations
 * \brief Create a random normal vector
 * \details Create a vector distributed according to normal distribution with mean
 *          0, variance 1.
 * 
 * \param g Computation graph
 * \param d The dimensions of the input
 * 
 * \return The new expression
 */
Expression random_normal(ComputationGraph& g, const Dim& d);

////////////////////////////////////////////////
// Arithmetic operations                      //
////////////////////////////////////////////////

// operators
Expression operator-(const Expression& x);
Expression operator+(const Expression& x, const Expression& y);
Expression operator+(const Expression& x, real y);
Expression operator+(real x, const Expression& y);
Expression operator-(const Expression& x, const Expression& y);
Expression operator-(real x, const Expression& y);
Expression operator-(const Expression& x, real y);
Expression operator*(const Expression& x, const Expression& y);
Expression operator*(const Expression& x, float y);
inline Expression operator*(float y, const Expression& x) { return x * y; }
inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }

template <typename T>
inline Expression affine_transform(const T& xs) { return detail::f<AffineTransform>(xs); }
inline Expression affine_transform(const std::initializer_list<Expression>& xs) { return detail::f<AffineTransform>(xs); }

Expression sqrt(const Expression& x);
Expression erf(const Expression& x);
Expression tanh(const Expression& x);
Expression exp(const Expression& x);
Expression square(const Expression& x);
Expression cube(const Expression& x);
Expression lgamma(const Expression& x);
Expression log(const Expression& x);
Expression logistic(const Expression& x);
Expression rectify(const Expression& x);

Expression softmax(const Expression& x);
Expression softsign(const Expression& x);

Expression pow(const Expression& x, const Expression& y);
Expression min(const Expression& x, const Expression& y);
Expression max(const Expression& x, const Expression& y);

Expression dot_product(const Expression& x, const Expression& y);

// colwise addition
Expression addmv(const Expression& M, const Expression& v);
// componentwise division
Expression cwise_multiply(const Expression& x, const Expression& y);
Expression cdiv(const Expression& x, const Expression& y);
Expression colwise_add(const Expression& x, const Expression& bias);

////////////////////////////////////////////////
// Probability/loss operations                //
////////////////////////////////////////////////

Expression hinge(const Expression& x, unsigned index, float m = 1.0);
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m = 1.0);
Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m = 1.0);
Expression log_softmax(const Expression& x);
Expression sparsemax(const Expression& x);
Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);
Expression sparsemax(const Expression& x);
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>& target_support);
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>* ptarget_support);

Expression pickneglogsoftmax(const Expression& x, unsigned v);
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> & v);
Expression pickneglogsoftmax(const Expression& x, unsigned * pv);
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> * pv);

Expression squared_norm(const Expression& x);
Expression squared_distance(const Expression& x, const Expression& y);
Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);
Expression l1_distance(const Expression& x, const Expression& y);
Expression binary_log_loss(const Expression& x, const Expression& y);
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m=1.0);
Expression poisson_loss(const Expression& x, unsigned y);
Expression poisson_loss(const Expression& x, const unsigned* py);

template <typename T>
inline Expression logsumexp(const T& xs) { return detail::f<LogSumExp>(xs); }
inline Expression logsumexp(const std::initializer_list<Expression>& xs) { return detail::f<LogSumExp>(xs); }

template <typename T>
inline Expression sum(const T& xs) { return detail::f<Sum>(xs); }
inline Expression sum(const std::initializer_list<Expression>& xs) { return detail::f<Sum>(xs); }

template <typename T>
inline Expression max(const T& xs) { return detail::f<Max>(xs); }
inline Expression max(const std::initializer_list<Expression>& xs) { return detail::f<Max>(xs); }

template <typename T>
inline Expression average(const T& xs) { return detail::f<Average>(xs); }
inline Expression average(const std::initializer_list<Expression>& xs) { return detail::f<Average>(xs); }

////////////////////////////////////////////////
// Flow operations                            //
////////////////////////////////////////////////

/**
 * \ingroup flowoperations
 * \brief Prevent backprop
 * \details This node has no effect on the forward pass, but prevents gradients from
 *          flowing backward during the backward pass. This is useful when there's
 *          a subgraph for which you don't want loss passed back to the parameters.
 * 
 * \param x The input expression
 * 
 * \return The new expression
 */
Expression nobackprop(const Expression& x);

// reshape::forward is O(1), but backward is O(n)
Expression reshape(const Expression& x, const Dim& d);
// transpose requires O(n)
Expression transpose(const Expression& x);
Expression select_rows(const Expression& x, const std::vector<unsigned>& rows);
Expression select_rows(const Expression& x, const std::vector<unsigned>* prows);
// select_cols is more efficient than select_rows since Eigen uses column-major order
Expression select_cols(const Expression& x, const std::vector<unsigned>& cols);
Expression select_cols(const Expression& x, const std::vector<unsigned>* pcols);

// Sum the results of multiple batches
Expression sum_batches(const Expression& x);

// pick parts out of bigger objects
Expression pick(const Expression& x, unsigned v);
Expression pick(const Expression& x, const std::vector<unsigned> & v);
Expression pick(const Expression& x, unsigned * pv);
Expression pick(const Expression& x, const std::vector<unsigned> * pv);
Expression pickrange(const Expression& x, unsigned v, unsigned u);

template <typename T>
inline Expression concatenate_cols(const T& xs) { return detail::f<ConcatenateColumns>(xs); }
inline Expression concatenate_cols(const std::initializer_list<Expression>& xs) { return detail::f<ConcatenateColumns>(xs); }

template <typename T>
inline Expression concatenate(const T& xs) { return detail::f<Concatenate>(xs); }
inline Expression concatenate(const std::initializer_list<Expression>& xs) { return detail::f<Concatenate>(xs); }

////////////////////////////////////////////////
// Noise operations                           //
////////////////////////////////////////////////

Expression noise(const Expression& x, real stddev);
Expression dropout(const Expression& x, real p);
Expression block_dropout(const Expression& x, real p);

////////////////////////////////////////////////
// Convolution operations                     //
////////////////////////////////////////////////

Expression conv1d_narrow(const Expression& x, const Expression& f);
Expression conv1d_wide(const Expression& x, const Expression& f);
Expression filter1d_narrow(const Expression& x, const Expression& f);
Expression kmax_pooling(const Expression& x, unsigned k);
Expression fold_rows(const Expression& x, unsigned nrows=2);
Expression sum_cols(const Expression& x);
Expression average_cols(const Expression& x);
Expression kmh_ngram(const Expression& x, unsigned n);

////////////////////////////////////////////////
// Tensor operations                          //
////////////////////////////////////////////////

// z_ij = x_ijk * y_k
Expression contract3d_1d(const Expression& x, const Expression& y);
// z_i = x_ijk * y_k * z_j (+ b_i)
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z);
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b);
// z_ij = x_ijk * y_k + b_ij
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);


////////////////////////////////////////////////
// Linear algebra operations                  //
////////////////////////////////////////////////

// matrix inverse
Expression inverse(const Expression& x);
Expression logdet(const Expression& x);

Expression trace_of_product(const Expression& x, const Expression& y);


} }

#endif
