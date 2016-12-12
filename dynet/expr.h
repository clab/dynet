/**
 * \file expr.h
 * \defgroup operations operations
 * \defgroup inputoperations inputoperations
 * \defgroup arithmeticoperations arithmeticoperations
 * \defgroup lossoperations lossoperations
 * \defgroup flowoperations flowoperations
 * \defgroup noiseoperations noiseoperations
 * \defgroup convoperations convoperations
 * \defgroup tensoroperations tensoroperations
 * \defgroup linalgoperations linalgoperations
 * \brief The various operations that you can use in building a DyNet graph
 * 
 * \details TODO: **This documentation is incomplete. See expr.h for a full list of expressions.**
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
  const Dim& dim() const { return pg->get_dimension(i); }
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
 *          input. The dimensions of the input are defined by ``d``. So for example
 *          > ``input(g,{50},data)``: will result in a 50-length vector
 *          > ``input(g,{50,30},data)``: will result in a 50x30 matrix
 *          and so on, for an arbitrary number of dimensions.
 *          This function can also be used to import minibatched inputs. For example,
 *          if we have 10 examples in a minibatch, each with size 50x30, then we call
 *          > ``input(g,Dim({50,30},10),data)``
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
 * \param pindices Pointer to lookup indices
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

/**
 * \ingroup inputoperations
 * \brief Create a random bernoulli vector
 * \details Create a vector distributed according to bernoulli distribution with parameter p.
 * 
 * \param g Computation graph
 * \param d The dimensions of the input
 * \param p The bernoulli p parameter
 * \param scale A scaling factor for the output ("active" elements will receive this value)
 * 
 * \return A "d" dimensioned bernoulli distributed vector
 */
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale = 1.0f);

/**
 * \ingroup inputoperations
 * \brief Create a random uniform vector
 * \details Create a vector distributed according to uniform distribution with boundaries left and right.
 * 
 * \param g Computation graph
 * \param d The dimensions of the input
 * \param left The left boundary
 * \param right The right boundary
 * 
 * \return A "d" dimensioned uniform distributed vector
 */
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right);

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
 *          held in the input initializer list xs.
 *          The first expression is the "bias," which is added to the expression as-is.
 *          The remaining expressions are multiplied together in pairs, then added.
 *          A very common usage case is the calculation of the score for a neural network
 *          layer (e.g. b + Wz) where b is the bias, W is the weight matrix, and z is the
 *          input. In this case xs[0] = b, xs[1] = W, and xs[2] = z.
 * 
 * \param xs An initializer list containing an odd number of expressions
 * 
 * \return An expression equal to: xs[0] + xs[1]*xs[2] + xs[3]*xs[4] + ...
 */
inline Expression affine_transform(const std::initializer_list<Expression>& xs) { return detail::f<AffineTransform>(xs); }
template <typename T>
inline Expression affine_transform(const T& xs) { return detail::f<AffineTransform>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Sum
 * \details This performs an elementwise sum over all the expressions in xs
 * 
 * \param xs An initializer list containing expressions
 * 
 * \return An expression where the ith element is equal to xs[0][i] + xs[1][i] + ...
 */
inline Expression sum(const std::initializer_list<Expression>& xs) { return detail::f<Sum>(xs); }
template <typename T>
inline Expression sum(const T& xs) { return detail::f<Sum>(xs); }

/**
 * \ingroup arithmeticoperations
 * \brief Average
 * \details This performs an elementwise average over all the expressions in xs
 * 
 * \param xs An initializer list containing expressions
 * 
 * \return An expression where the ith element is equal to (xs[0][i] + xs[1][i] + ...)/|xs|
 */
inline Expression average(const std::initializer_list<Expression>& xs) { return detail::f<Average>(xs); }
template <typename T>
inline Expression average(const T& xs) { return detail::f<Average>(xs); }

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
 * \brief Gaussian error function
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
 * \return An expression where the ith element is equal to e^{x_i}
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
 * \return An expression where the ith element is equal to y_i = 1/(1+e^{x_i})
 */
Expression logistic(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Rectifier
 * \details Calculate elementwise the recitifer (ReLU) function y_i = max(x_i,0)
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
 * \return An expression where the ith element is equal to x_i/(1+|x_i|)
 */
Expression softsign(const Expression& x);

/**
 * \ingroup arithmeticoperations
 * \brief Power function
 * \details Calculate an output where the ith element is equal to x_i^y_i
 * 
 * \param x The input expression
 * \param y The exponent expression
 * 
 * \return An expression where the ith element is equal to x_i^y_i
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
 * \return An expression where the ith element is equal to min(x_i,y_i)
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
 * \brief Max
 * \details This performs an elementwise max over all the expressions in xs
 * 
 * \param xs An initializer list containing expressions
 * 
 * \return An expression where the ith element is equal to max(xs[0][i], xs[1][i], ...)
 */
inline Expression max(const std::initializer_list<Expression>& xs) { return detail::f<Max>(xs); }
template <typename T>
inline Expression max(const T& xs) { return detail::f<Max>(xs); }

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
 * \return An expression where the ith element is equal to x_i*y_i
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
 * \return An expression where the ith element is equal to x_i/y_i
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
 * \return An expression where bias is added to each column of x
 */
Expression colwise_add(const Expression& x, const Expression& bias);

////////////////////////////////////////////////
// Probability/loss operations                //
////////////////////////////////////////////////

/**
 * \ingroup lossoperations
 * \brief Softmax
 * \details The softmax function, which sets each element to be e^{x[i]}/{sum_j e^{x[j]}}.
 * 
 * \param x A vector
 * 
 * \return A vector after calculating the softmax
 */
Expression softmax(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Log softmax
 * \details The log of the softmax function, which sets each element to be 
 *          log( e^{x[i]}/{sum_j e^{x[j]}} ).
 * 
 * \param x A vector
 * 
 * \return A vector after calculating the log softmax
 */
Expression log_softmax(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Restricted log softmax
 * \details The log softmax function calculated over only a subset of the vector elements. The
 *          elements to be included are set by the ``restriction`` variable. All elements not
 *          included in ``restriction`` are set to negative infinity.
 * 
 * \param x A vector over which to calculate the softmax
 * \param restriction The elements over which to calculate the softmax
 * 
 * \return A vector with the log softmax over the specified elements
 */
Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);

/**
 * \ingroup lossoperations
 * \brief Log, sum, exp
 * \details The elementwise "logsumexp" function that calculates
 *   \f$ln(\sum_i e^{xs_i})\f$, used in adding probabilities in the log domain.
 * 
 * \param xs Expressions with respect to which to calculate the logsumexp.
 * 
 * \return The result.
 */
inline Expression logsumexp(const std::initializer_list<Expression>& xs) { return detail::f<LogSumExp>(xs); }
template <typename T>
inline Expression logsumexp(const T& xs) { return detail::f<LogSumExp>(xs); }

/**
 * \ingroup lossoperations
 * \brief Negative softmax log likelihood
 * \details This function takes in a vector of scores ``x``, and performs a log softmax, takes
 *          the negative, and selects the likelihood corresponding to the element ``v``. This is
 *          perhaps the most standard loss function for training neural networks to predict
 *          one out of a set of elements.
 * 
 * \param x A vector of scores
 * \param v The element with which to calculate the loss
 * 
 * \return The negative log likelihood of element ``v`` after taking the softmax
 */
Expression pickneglogsoftmax(const Expression& x, unsigned v);

/**
 * \ingroup lossoperations
 * \brief Modifiable negative softmax log likelihood
 * \details This function calculates the negative log likelihood after the softmax with
 *          respect to index ``*pv``. This computes the same value as the previous function
 *          that passes the index ``v`` by value, but instead passes by pointer so the value
 *          ``*pv`` can be modified without re-constructing the computation graph. This can be
 *          used in situations where we want to create a computation graph once, then feed it
 *          different data points.
 * 
 * \param x A vector of scores
 * \param pv A pointer to the index of the correct element
 * 
 * \return The negative log likelihood of element ``*pv`` after taking the softmax
 */
Expression pickneglogsoftmax(const Expression& x, unsigned * pv);

/**
 * \ingroup lossoperations
 * \brief Batched negative softmax log likelihood
 * \details This function is similar to standard pickneglogsoftmax, but calculates loss with
 *          respect to multiple batch elements. The input will be a mini-batch of score vectors
 *          where the number of batch elements is equal to the number of indices in ``v``.
 * 
 * \param x An expression with vectors of scores over N batch elements
 * \param v A size-N vector indicating the index with respect to all the batch elements
 * 
 * \return The negative log likelihoods over all the batch elements
 */
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> & v);

/**
 * \ingroup lossoperations
 * \brief Modifiable batched negative softmax log likelihood
 * \details This function is a combination of modifiable pickneglogsoftmax and batched
 *          pickneglogsoftmax: ``pv`` can be modified without re-creating the computation graph.
 * 
 * \param x An expression with vectors of scores over N batch elements
 * \param pv A pointer to the indexes
 * 
 * \return The negative log likelihoods over all the batch elements
 */
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> * pv);

/**
 * \ingroup lossoperations
 * \brief Hinge loss
 * \details This expression calculates the hinge loss, formally expressed as:
 *          \f$ \text{hinge}(x,index,m) = \sum_{i \ne index} \max(0, m-x[index]+x[i]). \f$
 * 
 * \param x A vector of scores
 * \param index The index of the correct candidate
 * \param m The margin
 * 
 * \return The hinge loss of candidate ``index`` with respect to margin ``m``
 */
Expression hinge(const Expression& x, unsigned index, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Modifiable hinge loss
 * \details This function calculates the hinge loss with
 *          with respect to index ``*pindex``. This computes the same value as the previous function
 *          that passes the index ``index`` by value, but instead passes by pointer so the value
 *          ``*pindex`` can be modified without re-constructing the computation graph. This can be
 *          used in situations where we want to create a computation graph once, then feed it
 *          different data points.
 * 
 * \param x A vector of scores
 * \param pindex A pointer to the index of the correct candidate
 * \param m The margin
 * 
 * \return The hinge loss of candidate ``*pindex`` with respect to margin ``m``
 */
Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Batched hinge loss
 * \details The same as hinge loss, but for the case where ``x`` is a mini-batched tensor
 *          with ``indices.size()`` batch elements, and ``indices`` is a vector indicating
 *          the index of each of the correct elements for these elements.
 * 
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param indices The indices of the correct candidates for each batch element
 * \param m The margin
 * 
 * \return The hinge loss of each mini-batch
 */
Expression hinge(const Expression& x, const std::vector<unsigned> & indices, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Batched modifiable hinge loss
 * \details A combination of the previous batched and modifiable hinge loss functions, where
 *          vector ``*pindices`` can be modified.
 * 
 * \param x A mini-batch of vectors with ``indices.size()`` batch elements
 * \param pindices Pointer to the indices of the correct candidates for each batch element
 * \param m The margin
 * 
 * \return The hinge loss of each mini-batch
 */
Expression hinge(const Expression& x, const std::vector<unsigned> * pindices, float m = 1.0);

/**
 * \ingroup lossoperations
 * \brief Sparsemax
 * \details The sparsemax function (Martins et al. 2016), which is similar to softmax,
 *          but induces sparse solutions where most of the vector elements are zero.
 *          **Note:** This function is not yet implemented on GPU.
 * 
 * \param x A vector of scores
 * 
 * \return The sparsemax of the scores
 */
Expression sparsemax(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Sparsemax loss
 * \details The sparsemax loss function (Martins et al. 2016), which is similar to
 *          softmax loss, but induces sparse solutions where most of the vector
 *          elements are zero. It has a gradient similar to the sparsemax function
 *          and thus is useful for optimizing when the sparsemax will be used at
 *          test time.
 *          **Note:** This function is not yet implemented on GPU.
 * 
 * \param x A vector of scores
 * \param target_support The target correct labels.
 * 
 * \return The sparsemax loss of the labels
 */
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>& target_support);

/**
 * \ingroup lossoperations
 * \brief Modifiable sparsemax loss
 * \details Similar to the sparsemax loss, but with ptarget_support being a pointer
 *          to a vector, allowing it to be modified without re-creating the compuation
 *          graph.
 *          **Note:** This function is not yet implemented on GPU.
 * 
 * \param x A vector of scores
 * \param ptarget_support A pointer to the target correct labels.
 * 
 * \return The sparsemax loss of the labels
 */
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>* ptarget_support);

/**
 * \ingroup lossoperations
 * \brief Squared norm
 * \details The squared norm of the values of x: \f$\sum_i x_i^2\f$.
 * 
 * \param x A vector of values
 * 
 * \return The squared norm
 */
Expression squared_norm(const Expression& x);

/**
 * \ingroup lossoperations
 * \brief Squared distance
 * \details The squared distance between values of ``x`` and ``y``: \f$\sum_i (x_i-y_i)^2\f$.
 * 
 * \param x A vector of values
 * \param y Another vector of values
 * 
 * \return The squared distance
 */
Expression squared_distance(const Expression& x, const Expression& y);

/**
 * \ingroup lossoperations
 * \brief Squared distance
 * \details The L1 distance between values of ``x`` and ``y``: \f$\sum_i |x_i-y_i|\f$.
 * 
 * \param x A vector of values
 * \param y Another vector of values
 * 
 * \return The squared distance
 */
Expression l1_distance(const Expression& x, const Expression& y);

/**
 * \ingroup lossoperations
 * \brief Huber distance
 * \details The huber distance between values of ``x`` and ``y`` parameterized
 *    by ``c,`` \f$\sum_i L_c(x_i, y_i)\f$ where:
 *    
 *    \f$
 *      L_c(x, y) = \begin{cases}{lr}
 *        \frac{1}{2}(y - x)^2                   & \textrm{for } |y - f(x)| \le c, \\
 *        c\, |y - f(x)| - \frac{1}{2}c^2 & \textrm{otherwise.}
 *      \end{cases}
 *    \f$
 *
 * \param x A vector of values
 * \param y Another vector of values
 * \param c The parameter of the huber distance parameterizing the cuttoff
 * 
 * \return The huber distance
 */
Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);

/**
 * \ingroup lossoperations
 * \brief Binary log loss
 * \details The log loss of a binary decision according to the sigmoid
 *          sigmoid function \f$- \sum_i (y_i * ln(x_i) + (1-y_i) * ln(1-x_i)) \f$
 * 
 * \param x A vector of values
 * \param y A vector of true answers
 * 
 * \return The log loss of the sigmoid function
 */
Expression binary_log_loss(const Expression& x, const Expression& y);

/**
 * \ingroup lossoperations
 * \brief Pairwise rank loss
 * \details A margin-based loss, where every margin violation for each pair of
 *          values is penalized: \f$\sum_i max(x_i-y_i+m, 0)\f$
 * 
 * \param x A vector of values
 * \param y A vector of true answers
 * \param m The margin
 * 
 * \return The pairwise rank loss
 */
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m=1.0);

/**
 * \ingroup lossoperations
 * \brief Poisson loss
 * \details The negative log probability of ``y`` according to a Poisson
 *          distribution with parameter ``x``. Useful in Poisson regression
 *          where, we try to predict the parameters of a Possion distribution
 *          to maximize the probability of data ``y``.
 * 
 * \param x The parameter of the Poisson distribution.
 * \param y The target value
 * 
 * \return The Poisson loss
 */
Expression poisson_loss(const Expression& x, unsigned y);
/**
 * \ingroup lossoperations
 * \brief Modifiable Poisson loss
 * \details Similar to Poisson loss, but with the target value passed by
 *          pointer so that it can be modified without re-constructing the
 *          computation graph.
 * 
 * \param x The parameter of the Poisson distribution.
 * \param py A pointer to the target value
 * 
 * \return The Poisson loss
 */
Expression poisson_loss(const Expression& x, const unsigned* py);

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

/**
 * \ingroup flowoperations
 * \brief Reshape to another size
 * \details This node reshapes a tensor to another size, without changing the
 *          underlying layout of the data. The layout of the data in DyNet is
 *          column-major, so if we have a 3x4 matrix
 *    
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\
 *        x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\
 *        x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\
 *      \end{pmatrix}
 *    \f$
 *
 *          and transform it into a 2x6 matrix, it will be rearranged as:
 *
 *    \f$
 *      \begin{pmatrix}
 *        x_{1,1} & x_{3,1} & x_{2,2} & x_{1,3} & x_{3,3} & x_{2,4} \\
 *        x_{1,2} & x_{1,2} & x_{3,2} & x_{2,3} & x_{1,4} & x_{3,4} \\
 *      \end{pmatrix}
 *    \f$
 *
 *         **Note:** This is O(1) for forward, and O(n) for backward.
 * 
 * \param x The input expression
 * \param d The new dimensions
 * 
 * \return The reshaped expression
 */
Expression reshape(const Expression& x, const Dim& d);

/**
 * \ingroup flowoperations
 * \brief Transpose a matrix
 * \details Get the transpose of the matrix.
 *          **Note:** This is O(1) if either the row or column dimension is 1,
 *          and O(n) otherwise.
 * 
 * \param x The input expression
 * 
 * \return The transposed expression
 */
Expression transpose(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Select rows
 * \details Select a subset of rows of a matrix.
 * 
 * \param x The input expression
 * \param rows The rows to extract
 * 
 * \return An expression containing the selected rows
 */
Expression select_rows(const Expression& x, const std::vector<unsigned>& rows);

/**
 * \ingroup flowoperations
 * \brief Modifiable select rows
 * \details Select a subset of rows of a matrix, where the elements of prows
 *          can be modified without re-creating the computation graph.
 * 
 * \param x The input expression
 * \param prows The rows to extract
 * 
 * \return An expression containing the selected rows
 */
Expression select_rows(const Expression& x, const std::vector<unsigned>* prows);

/**
 * \ingroup flowoperations
 * \brief Select columns
 * \details Select a subset of columns of a matrix. select_cols is more 
 *          efficient than select_rows since DyNet uses column-major order.
 * 
 * \param x The input expression
 * \param columns The columns to extract
 * 
 * \return An expression containing the selected columns
 */
Expression select_cols(const Expression& x, const std::vector<unsigned>& cols);

/**
 * \ingroup flowoperations
 * \brief Modifiable select columns
 * \details Select a subset of columns of a matrix, where the elements of pcols
 *          can be modified without re-creating the computation graph.
 * 
 * \param x The input expression
 * \param pcolumns The columns to extract
 * 
 * \return An expression containing the selected columns
 */
Expression select_cols(const Expression& x, const std::vector<unsigned>* pcols);

/**
 * \ingroup flowoperations
 * \brief Sum over minibatches
 * \details Sum an expression that consists of multiple minibatches into one of
 *          equal dimension but with only a single minibatch. This is useful
 *          for summing loss functions at the end of minibatch training.
 * 
 * \param x The input mini-batched expression
 * 
 * \return An expression with a single batch
 */
Expression sum_batches(const Expression& x);

/**
 * \ingroup flowoperations
 * \brief Pick element
 * \details Pick a single element from an expression.
 * 
 * \param x The input expression
 * \param v The index of the element to select
 * 
 * \return The value of x[v]
 */
Expression pick(const Expression& x, unsigned v);

/**
 * \ingroup flowoperations
 * \brief Pick multiple elements
 * \details Pick multiple elements from an input expression
 * 
 * \param x The input expression
 * \param v A vector of indicies to choose
 * 
 * \return A vector of values {x[v[0]], x[v[1]], ...}
 */
Expression pick(const Expression& x, const std::vector<unsigned> & v);

/**
 * \ingroup flowoperations
 * \brief Modifiable pick element
 * \details Pick a single element from an expression, where the index is
 *          passed by pointer so we do not need to re-create the computation
 *          graph every time.
 * 
 * \param x The input expression
 * \param pv Pointer to the index of the element to select
 * 
 * \return The value of x[*pv]
 */
Expression pick(const Expression& x, unsigned * pv);

/**
 * \ingroup flowoperations
 * \brief Modifiable pick multiple elements
 * \details Pick multiple elements from an input expression, where the indices
 *          are passed by pointer so we do not need to re-create the computation
 *          graph every time.
 * 
 * \param x The input expression
 * \param pv A pointer to vector of indicies to choose
 * 
 * \return A vector of values {x[(*pv)[0]], x[(*pv)[1]], ...}
 */
Expression pick(const Expression& x, const std::vector<unsigned> * pv);

/**
 * \ingroup flowoperations
 * \brief Pick range of elements
 * \details Pick a range of elements from an expression.
 * 
 * \param x The input expression
 * \param v The beginning index
 * \param u The end index
 * 
 * \return The value of {x[v],...,x[u]}
 */
Expression pickrange(const Expression& x, unsigned v, unsigned u);

/**
 * \ingroup flowoperations
 * \brief Concatenate columns
 * \details Perform a concatenation of the columns in multiple expressions.
 *          All expressions must have the same number of rows.
 * 
 * \param xs The input expressions
 * 
 * \return The expression with the columns concatenated
 */
inline Expression concatenate_cols(const std::initializer_list<Expression>& xs) { return detail::f<ConcatenateColumns>(xs); }
template <typename T>
inline Expression concatenate_cols(const T& xs) { return detail::f<ConcatenateColumns>(xs); }

/**
 * \ingroup flowoperations
 * \brief Concatenate rows
 * \details Perform a concatenation of the rows in multiple expressions.
 *          All expressions must have the same number of columns.
 * 
 * \param xs The input expressions
 * 
 * \return The expression with the rows concatenated
 */
inline Expression concatenate(const std::initializer_list<Expression>& xs) { return detail::f<Concatenate>(xs); }
template <typename T>
inline Expression concatenate(const T& xs) { return detail::f<Concatenate>(xs); }

////////////////////////////////////////////////
// Noise operations                           //
////////////////////////////////////////////////

/**
 * \ingroup noiseoperations
 * \brief Gaussian noise
 * \details Add gaussian noise to an expression.
 * 
 * \param x The input expression
 * \param stddev The standard deviation of the gaussian
 * 
 * \return The noised expression
 */
Expression noise(const Expression& x, real stddev);

/**
 * \ingroup noiseoperations
 * \brief Dropout
 * \details
 *   With a fixed probability, drop out (set to zero) nodes in the input
 *   expression, and **scale** the remaining nodes by 1/p. Note that there are
 *   [two kinds of dropout](http://cs231n.github.io/neural-networks-2/#reg):
 *   - *Regular dropout:* where we perform dropout at training time and then\n
 *     scale outputs by p at test time.
 *   - *Inverted dropout:* where we perform dropout and scaling at training\n
 *     time, and do not need to do anything at test time.
 *   DyNet implements the latter, so you only need to apply dropout at training
 *   time, and do not need to perform scaling and test time.
 * 
 * \param x The input expression
 * \param p The dropout probability
 * 
 * \return The dropped out expression
 */
Expression dropout(const Expression& x, real p);

/**
 * \ingroup noiseoperations
 * \brief Block dropout
 * \details Identical to the dropout operation, but either drops out *all*
 *          or *no* values in the expression, as opposed to making a decision
 *          about each value individually.
 * 
 * \param x The input expression
 * \param p The block dropout probability
 * 
 * \return The block dropout expression
 */
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
