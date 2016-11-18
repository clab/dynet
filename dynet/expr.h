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
 * \return An expression representing s
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
 * \return An expression representing *ps
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
 * \return An expression representing data
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
 * \param pdata A pointer to an (updatable) vector of data points
 * 
 * \return An expression representing *pdata
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
 * \return An expression representing data
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
 * \return An expression representing p
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
 * \return An expression representing the constant p
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
 * \return An expression representing p[index]
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
 * \return An expression representing p[*pindex]
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
 * \return A constant expression representing p[index]
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
 * \return A constant expression representing p[*pindex]
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
 * \return An expression with the "i"th batch element representing p[indices[i]]
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
 * \return An expression with the "i"th batch element representing p[*pindices[i]]
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
 * \return A constant expression with the "i"th batch element representing p[indices[i]]
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
 * \return A constant expression with the "i"th batch element representing
 *         p[*pindices[i]]
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
 * \return A "d" dimensioned zero vector
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
 * \return A "d" dimensioned normally distributed vector
 */
Expression random_normal(ComputationGraph& g, const Dim& d);

////////////////////////////////////////////////
// Arithmetic operations                      //
////////////////////////////////////////////////

/**
 * \ingroup arithmeticoperations
 * \brief Negation
 * \details Negate the passed argument.
 * 
 * \param x An input expression
 * 
 * \return The negation of x
 */
Expression operator-(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Expression addition
 * \details Add two expressions of the same dimensions.
 * 
 * \param x The first input
 * \param y The second input
 * 
 * \return The sum of x and y
 */
Expression operator+(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar addition
 * \details Add a scalar to an expression
 * 
 * \param x The expression
 * \param y The scalar
 * 
 * \return An expression equal to x, with every component increased by y
 */
Expression operator+(const Expression& x, real y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar addition
 * \details Add a scalar to an expression
 * 
 * \param x The scalar
 * \param y The expression
 * 
 * \return An expression equal to y, with every component increased by x
 */
Expression operator+(real x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Expression subtraction
 * \details Subtract one expression from another.
 * 
 * \param x The expression from which to subtract
 * \param y The expression to subtract
 * 
 * \return An expression where the ith element is x_i minus y_i
 */
Expression operator-(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar subtraction
 * \details Subtract an expression from a scalar
 * 
 * \param x The scalar from which to subtract
 * \param y The expression to subtract
 * 
 * \return An expression where the ith element is x_i minus y
 */
Expression operator-(real x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Scalar subtraction
 * \details Subtract a scalar from an expression
 * 
 * \param x The expression from which to subtract
 * \param y The scalar to subtract
 * 
 * \return An expression where the ith element is x_i minus y
 */
Expression operator-(const Expression& x, real y);


/**
 * \ingroup arithmeticoperations
 * \brief Matrix multiplication
 * \details Multiply two matrices together. Like standard matrix multiplication, the
 *          second dimension of x and the first dimension of y must match.
 * 
 * \param x The left-hand matrix
 * \param y The right-hand matrix
 * 
 * \return An expression x times y
 */
Expression operator*(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Matrix-scalar multiplication
 * \details Multiply an expression component-wise by a scalar.
 * 
 * \param x The matrix
 * \param y The scalar
 * 
 * \return An expression where the ith element is x_i times y
 */
Expression operator*(const Expression& x, float y);

/**
 * \ingroup arithmeticoperations
 * \brief Matrix-scalar multiplication
 * \details Multiply an expression component-wise by a scalar.
 * 
 * \param x The scalar
 * \param y The matrix
 * 
 * \return An expression where the ith element is x_i times y
 */
inline Expression operator*(float y, const Expression& x) { return x * y; }

/**
 * \ingroup arithmeticoperations
 * \brief Matrix-scalar division
 * \details Divide an expression component-wise by a scalar.
 * 
 * \param x The matrix
 * \param y The scalar
 * 
 * \return An expression where the ith element is x_i divided by y
 */
inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }

/**
 * \ingroup arithmeticoperations
 * \brief Affine transform
 * \details This performs an affine transform over an arbitrary (odd) number of expressions
 *          held in the input initializer list x.
 *          The first expression is the "bias," which is added to the expression as-is.
 *          The remaining expressions are multiplied together, then added.
 *          A very common usage case is the calculation of the score for a neural network
 *          layer (e.g. b + Wz) where b is the bias, W is the weight matrix, and z is the
 *          input. In this case x[0] = b, x[1] = W, and x[2] = z.
 * 
 * \param x An initializer list containing an odd number of expressions
 * 
 * \return An expression equal to: x[0] + x[1]*x[2] + x[3]*x[4] + ...
 */
inline Expression affine_transform(const std::initializer_list<Expression>& xs) { return detail::f<AffineTransform>(xs); }
template <typename T>
inline Expression affine_transform(const T& xs) { return detail::f<AffineTransform>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Square root
 * \details Elementwise square root.
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to sqrt(x_i)
 */
Expression sqrt(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Gaussian errror function
 * \details Elementwise calculation of the Gaussian error function
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to erf(x_i)
 */
Expression erf(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Hyperbolic tangent
 * \details Elementwise calculation of the hyperbolic tangent
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to tanh(x_i)
 */
Expression tanh(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Natural exponent
 * \details Calculate elementwise y_i = e^{x_i}
 * 
 * \param x The input expression
 * 
 * \return An expression equal to e^{x_i}
 */
Expression exp(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Square
 * \details Calculate elementwise y_i = x_i^2
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to x_i^2
 */
Expression square(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Cube
 * \details Calculate elementwise y_i = x_i^3
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to x_i^3
 */
Expression cube(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Log gamma
 * \details Calculate elementwise y_i = ln(gamma(x_i))
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to ln(gamma(x_i))
 */
Expression lgamma(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Logarithm
 * \details Calculate the elementwise natural logarithm y_i = ln(x_i)
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to ln(x_i)
 */
Expression log(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Logistic sigmoid function
 * \details Calculate elementwise y_i = 1/(1+e^{x_i})
 * 
 * \param x The input expression
 * 
 * \return An expression equal to y_i = 1/(1+e^{x_i})
 */
Expression logistic(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Rectifier
 * \details Calculate elementwise the recitifer (RelU) function y_i = max(x_i,0)
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is equal to max(x_i,0)
 */
Expression rectify(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Soft Sign
 * \details Calculate elementwise the softsign function y_i = x_i/(1+|x_i|)
 * 
 * \param x The input expression
 * 
 * \return An expression equal to x_i/(1+|x_i|)
 */
Expression softsign(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Power function
 * \details Calculate an output where the ith element is equal to x_i^y_i
 * 
 * \param x The input expression
 * 
 * \return An expression where the ith element is to x_i^y_i
 */
Expression pow(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Minimum
 * \details Calculate an output where the ith element is min(x_i,y_i)
 * 
 * \param x The first input expression
 * \param y The second input expression
 * 
 * \return An expression equal to min(x_i,y_i)
 */
Expression min(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Maximum
 * \details Calculate an output where the ith element is max(x_i,y_i)
 * 
 * \param x The first input expression
 * \param y The second input expression
 * 
 * \return An expression where the ith element is equal to max(x_i,y_i)
 */
Expression max(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Dot Product
 * \details Calculate the dot product sum_i x_i*y_i
 * 
 * \param x The input expression
 * \param y The input expression
 * 
 * \return An expression equal to the dot product
 */
Expression dot_product(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Componentwise multiply
 * \details Do a componentwise multiply where each value is equal to x_i*y_i.
 *          This function used to be called cwise_multiply.
 * 
 * \param x The first input expression
 * \param y The second input expression
 * 
 * \return An expression where the ith element is x_i*y_i 
 */
Expression cmult(const Expression& x, const Expression& y);


/**
 * \ingroup arithmeticoperations
 * \brief Componentwise multiply
 * \details Do a componentwise multiply where each value is equal to x_i/y_i
 * 
 * \param x The first input expression
 * \param y The second input expression
 * 
 * \return An expression where the ith element is x_i/y_i 
 */
Expression cdiv(const Expression& x, const Expression& y);

/**
 * \ingroup arithmeticoperations
 * \brief Columnwise addition
 * \details Add vector "bias" to each column of matrix "x"
 * 
 * \param x An MxN matrix
 * \param bias A length M vector
 * 
 * \return An expression bias is added to each column of x
 */
Expression colwise_add(const Expression& x, const Expression& bias);

////////////////////////////////////////////////
// Probability/loss operations                //
////////////////////////////////////////////////

Expression hinge(const Expression& x, unsigned index, float m = 1.0);
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m = 1.0);
Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m = 1.0);
Expression softmax(const Expression& x);
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
