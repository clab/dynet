#ifndef DYNET_C_EXPR_H_
#define DYNET_C_EXPR_H_

#include <dynet_c/define.h>
#include <dynet_c/devices.h>
#include <dynet_c/dim.h>
#include <dynet_c/graph.h>
#include <dynet_c/model.h>
#include <dynet_c/tensor.h>

/**
 * Opaque type of Expression.
 */
typedef struct dynetExpression dynetExpression_t;

/**
 * Creates a new Expression object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateExpression(dynetExpression_t **newobj);

/**
 * Creates a clone of the existing Expression object.
 * @param src Pointer to a source Expression.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCloneExpression(
    const dynetExpression_t *src, dynetExpression_t **newobj);

/**
 * Deletes the Expression object.
 * @param expr Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteExpression(dynetExpression_t *expr);

/**
 * Returns dim of the parameter.
 * @param expr Pointer of a handler.
 * @param newobj Pointer to receive a Dim object.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetExpressionDim(
    const dynetExpression_t *expr, const dynetDim_t **newobj);

/**
 * Gets value of the expression.
 * @param expr Pointer of a handler.
 * @param tensor Pointer to receive a tensor of the value.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetExpressionValue(
    const dynetExpression_t *expr, const dynetTensor_t **tensor);

/**
 * Gets gradient of the expression.
 * @param expr Pointer of a handler.
 * @param tensor Pointer to receive a tensor of the gradient.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetExpressionGradient(
    const dynetExpression_t *expr, const dynetTensor_t **tensor);

/**
 * Inputs scalar.
 * @param g Computation graph.
 * @param s Real number.
 * @param device The place device for the input value. If nullptr is given,
 *               default_device will be used instead.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyInputScalar(
    dynetComputationGraph_t *g, float s, dynetDevice_t *device,
    dynetExpression_t **newobj);

/**
 * Inputs vector/matrix/tensor.
 * @param g Computation graph.
 * @param d Dimension of the input matrix.
 * @param data A vector of data points.
 * @param n Number of values.
 * @param device The place device for the input value. If nullptr is given,
 *               default_device will be used instead.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyInput(
    dynetComputationGraph_t *g, const dynetDim_t *d, const float *data,
    size_t n, dynetDevice_t *device, dynetExpression_t **newobj);

/**
 * Inputs sparse vector.
 * @param g Computation graph.
 * @param d Dimension of the input matrix.
 * @param ids The indexes of the data points to update.
 * @param n_ids Number of ids.
 * @param data The data points corresponding to each index.
 * @param n_data Number of data points.
 * @param defdata The default data with which to set the unspecified data
 *                points.
 * @param device The place device for the input value. If nullptr is given,
 *               default_device will be used instead.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyInputSparse(
    dynetComputationGraph_t *g, const dynetDim_t *d, const uint32_t *ids,
    size_t n_ids, const float *data, size_t n_data, float defdata,
    dynetDevice_t *device, dynetExpression_t **newobj);

/**
 * Creates batched one hot vectors.
 * @param g Computation graph.
 * @param d Dimension of the input vector.
 * @param ids The indices we want to set to 1, one per batch element.
 * @param n Number of ids.
 * @param device The place device for the input value. If nullptr is given,
 *               default_device will be used instead.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyOneHot(
    dynetComputationGraph_t *g, uint32_t d, const uint32_t *ids, size_t n,
    dynetDevice_t *device, dynetExpression_t **newobj);

/**
 * Loads parameter.
 * @param g Computation graph.
 * @param p Parameter to load.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyParameter(
    dynetComputationGraph_t *g, dynetParameter_t *p,
    dynetExpression_t **newobj);

/**
 * Loads lookup parameter.
 * @param g Computation graph.
 * @param lp LookupParameter to load.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLookupParameter(
    dynetComputationGraph_t *g, dynetLookupParameter_t *lp,
    dynetExpression_t **newobj);

/**
 * Loads constant parameter.
 * @param g Computation graph.
 * @param p Parameter to load.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstParameter(
    dynetComputationGraph_t *g, const dynetParameter_t *p,
    dynetExpression_t **newobj);

/**
 * Loads constant lookup parameter.
 * @param g Computation graph.
 * @param p LookupParameter to load.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstLookupParameter(
    dynetComputationGraph_t *g, const dynetLookupParameter_t *p,
    dynetExpression_t **newobj);

/**
 * Looks up parameter.
 * @param g Computation graph.
 * @param p LookupParameter object from which to load.
 * @param index Index of the parameters within p.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLookupOne(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p, uint32_t index,
    dynetExpression_t **newobj);

/**
 * Looks up parameters.
 * @param g Computation graph.
 * @param p LookupParameter object from which to load.
 * @param indices Index of the parameters at each position in the batch.
 * @param n Number of indices.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLookup(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p,
    const uint32_t *indices, size_t n, dynetExpression_t **newobj);

/**
 * Looks up constant parameter.
 * @param g Computation graph.
 * @param p LookupParameter object from which to load.
 * @param index Index of the parameters within p.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstLookupOne(
    dynetComputationGraph_t *g, const dynetLookupParameter_t *p,
    uint32_t index, dynetExpression_t **newobj);

/**
 * Looks up constant parameters.
 * @param g Computation graph.
 * @param p LookupParameter object from which to load.
 * @param indices Index of the parameters at each position in the batch.
 * @param n Number of indices.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstLookup(
    dynetComputationGraph_t *g, const dynetLookupParameter_t *p,
    const uint32_t *indices, size_t n, dynetExpression_t **newobj);

/**
 * Creates an input full of zeros.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyZeros(
    dynetComputationGraph_t *g, const dynetDim_t *d,
    dynetExpression_t **newobj);

/**
 * Creates an input full of ones.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyOnes(
    dynetComputationGraph_t *g, const dynetDim_t *d,
    dynetExpression_t **newobj);

/**
 * Creates an input with one constant value.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param val Value of the input.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstant(
    dynetComputationGraph_t *g, const dynetDim_t *d, float val,
    dynetExpression_t **newobj);

/**
 * Creates a random normal vector.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param mean Mean of the distribution.
 * @param stddev Standard deviation of the distribution.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRandomNormal(
    dynetComputationGraph_t *g, const dynetDim_t *d, float mean, float stddev,
    dynetExpression_t **newobj);

/**
 * Creates a random bernoulli vector.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param p The bernoulli p parameter.
 * @param scale A scaling factor for the output ("active" elements will receive
 *              this value).
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRandomBernoulli(
    dynetComputationGraph_t *g, const dynetDim_t *d, float p, float scale,
    dynetExpression_t **newobj);

/**
 * Creates a random uniform vector.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param left The left boundary.
 * @param right The right boundary.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRandomUniform(
    dynetComputationGraph_t *g, const dynetDim_t *d, float left, float right,
    dynetExpression_t **newobj);

/**
 * Creates a random Gumbel sampled vector.
 * @param g Computation graph.
 * @param d Dimensions of the input.
 * @param mu The mu parameter.
 * @param beta The beta parameter.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRandomGumbel(
    dynetComputationGraph_t *g, const dynetDim_t *d, float mu, float beta,
    dynetExpression_t **newobj);

/**
 * Applies negation operation.
 * @param x An input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyNegative(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Applies addition operation.
 * @param x The first input expression.
 * @param y The second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAdd(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Applies addition operation.
 * @param x The input expression.
 * @param y The input scalar.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAddConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj);

/**
 * Applies addition operation.
 * @param x The input scalar.
 * @param y The input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAddExpr(
    float x, const dynetExpression_t *y, dynetExpression_t **newobj);

/**
 * Applies subtraction operation.
 * @param x The first input expression.
 * @param y The second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySubtract(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Applies subtraction operation.
 * @param x The input expression.
 * @param y The input scalar.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySubtractConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj);

/**
 * Applies subtraction operation.
 * @param x The input scalar.
 * @param y The input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySubtractExpr(
    float x, const dynetExpression_t *y, dynetExpression_t **newobj);

/**
 * Applies multiplication operation.
 * @param x The first input expression.
 * @param y The second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMultiply(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Applies multiplication operation.
 * @param x The input expression.
 * @param y The input scalar.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMultiplyConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj);

/**
 * Applies multiplication operation.
 * @param x The input scalar.
 * @param y The input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMultiplyExpr(
    float x, const dynetExpression_t *y, dynetExpression_t **newobj);

/**
 * Applies division operation.
 * @param x The first input expression.
 * @param y The second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyDivide(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Applies division operation.
 * @param x The input expression.
 * @param y The input scalar.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyDivideConst(
    const dynetExpression_t *x, float y, dynetExpression_t **newobj);

/**
 * Applies affine transform operation.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAffineTransform(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Applies sum operation.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySum(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Sums all elements.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySumElems(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes moment over all elements.
 * @param x Input mini-batched expression.
 * @param r Order of the moment.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMomentElems(
    const dynetExpression_t *x, uint32_t r, dynetExpression_t **newobj);

/**
 * Computes mean over all elements.
 * @param x Input mini-batched expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMeanElems(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes standard deviation over all elements.
 * @param x Input mini-batched expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyStdElems(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Applies sum over mini-batches.
 * @param x Input mini-batched expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySumBatches(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes moment over mini-batches.
 * @param x Input mini-batched expression.
 * @param r Order of the moment.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMomentBatches(
    const dynetExpression_t *x, uint32_t r, dynetExpression_t **newobj);

/**
 * Computes mean over mini-batches.
 * @param x Input mini-batched expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMeanBatches(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes standard deviation over mini-batches.
 * @param x Input mini-batched expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyStdBatches(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes sum along a specific dimension(s).
 * @param x Input mini-batched expression.
 * @param dims Dimensions along which to reduce.
 * @param n_dims Number of dims.
 * @param b Whether to include batch dimension.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySumDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    DYNET_C_BOOL b, dynetExpression_t **newobj);

/**
 * Computes cumulative sum along a specific dimension.
 * @param x Input mini-batched expression.
 * @param d Dimension along which to compute the cumulative sum.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCumsum(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj);

/**
 * Computes moment along a specific dimension.
 * @param x Input mini-batched expression.
 * @param dims Dimensions along which to reduce.
 * @param n_dims Number of dims.
 * @param r Order of the moment.
 * @param b Whether to include batch dimension.
 * @param n If > 0, overwrite the n in the equation by this value, useful for
 *          masking.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMomentDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    uint32_t r, DYNET_C_BOOL b, uint32_t n, dynetExpression_t **newobj);

/**
 * Computes mean along a specific dimension.
 * @param x Input mini-batched expression.
 * @param dims Dimensions along which to reduce.
 * @param n_dims Number of dims.
 * @param b Whether to include batch dimension.
 * @param n If > 0, overwrite the n in the equation by this value, useful for
 *          masking.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMeanDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    DYNET_C_BOOL b, uint32_t n, dynetExpression_t **newobj);

/**
 * Computes standard deviation along a specific dimension.
 * @param x Input mini-batched expression.
 * @param dims Dimensions along which to reduce.
 * @param n_dims Number of dims.
 * @param b Whether to include batch dimension.
 * @param n If > 0, overwrite the n in the equation by this value, useful for
 *          masking.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyStdDim(
    const dynetExpression_t *x, const uint32_t *dims, size_t n_dims,
    DYNET_C_BOOL b, uint32_t n, dynetExpression_t **newobj);

/**
 * Computes element-wise average over all expressions.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAverage(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Computes square root.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySqrt(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes absolute value.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAbs(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes the value of the Gaussian error function.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyErf(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes inverse sine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAsin(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes inverse cosine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAcos(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes inverse tangent.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAtan(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes sine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySin(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes cosine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCos(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes tangent.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyTan(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes hyperbolic sine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySinh(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes hyperbolic cosine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCosh(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes hyperbolic tangent.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyTanh(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes inverse hyperbolic sine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAsinh(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes inverse hyperbolic cosine.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAcosh(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes inverse hyperbolic tangent.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyAtanh(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes natural exponent.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyExp(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes square.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySquare(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes cube.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCube(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes log sigmoid.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLogSigmoid(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes log gamma.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLgamma(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes logarithm.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLog(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes logistic sigmoid.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLogistic(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes rectifier.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRectify(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes exponential linear unit.
 * @param x Input expression.
 * @param alpha The alpha value of the equation.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyElu(
    const dynetExpression_t *x, float alpha, dynetExpression_t **newobj);

/**
 * Computes scaled exponential linear unit.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySelu(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes SILU / SiL / Swish.
 * @param x Input expression.
 * @param beta The beta value of the equation.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySilu(
    const dynetExpression_t *x, float beta, dynetExpression_t **newobj);

/**
 * Computes soft sign.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySoftsign(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes power.
 * @param x Input expression.
 * @param y Exponent expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPow(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes binary minimum.
 * @param x First input expression.
 * @param y Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyBmin(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes binary maximum.
 * @param x First input expression.
 * @param y Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyBmax(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes maximum over all expressions.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMax(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Computes dot product.
 * @param x First input expression.
 * @param y Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyDotProduct(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes circular convolution.
 * @param u First input expression.
 * @param v Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCircConv(
    const dynetExpression_t *u, const dynetExpression_t *v,
    dynetExpression_t **newobj);

/**
 * Computes circular correlation.
 * @param u First input expression.
 * @param v Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCircCorr(
    const dynetExpression_t *u, const dynetExpression_t *v,
    dynetExpression_t **newobj);

/**
 * Computes componentwise multiplication.
 * @param x First input expression.
 * @param y Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCmult(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes componentwise division.
 * @param x First input expression.
 * @param y Second input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCdiv(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes columnwise addition.
 * @param x An MxN matrix.
 * @param bias A length M vector.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyColwiseAdd(
    const dynetExpression_t *x, const dynetExpression_t *bias,
    dynetExpression_t **newobj);

/**
 * Computes componentwise rounding.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRoundWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes componentwise rounding.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRoundWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes componentwise ceiling.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCeilWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes componentwise ceiling.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyCeilWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes componentwise floor.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyFloorWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes componentwise floor.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyFloorWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes softmax.
 * @param x A vector or matrix.
 * @param d Dimension to normalize over.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySoftmax(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj);

/**
 * Computes log softmax.
 * @param x A vector or matrix.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLogSoftmax(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes restricted log softmax.
 * @param x A vector or matrix.
 * @param restriction The elements over which to calculate the softmax.
 * @param n Number of restrictions.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyRestrictedLogSoftmax(
    const dynetExpression_t *x, const uint32_t *restriction, size_t n,
    dynetExpression_t **newobj);

/**
 * Computes log, sum, and exp by dimension.
 * @param x Expression with respect to which to calculate the logsumexp.
 * @param d The dimension along which to do the logsumexp.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLogsumexpDim(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj);

/**
 * Computes log, sum, and exp.
 * @param xs Expressions with respect to which to calculate the logsumexp.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLogsumexp(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Computes negative softmax log likelihood.
 * @param x A vector of scores.
 * @param v The element with which to calculate the loss.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPickneglogsoftmaxOne(
    const dynetExpression_t *x, uint32_t v, dynetExpression_t **newobj);

/**
 * Computes batched negative softmax log likelihood.
 * @param x An expression with vectors of scores over N batch elements.
 * @param v A size-N vector indicating the index with respect to all the batch
 *          elements.
 * @param n Number of indices.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPickneglogsoftmax(
    const dynetExpression_t *x, const uint32_t *v, size_t n,
    dynetExpression_t **newobj);

/**
 * Computes hinge loss.
 * @param x A vector of scores.
 * @param index The index of the correct candidate.
 * @param m The margin.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyHingeOne(
    const dynetExpression_t *x, uint32_t index, float m,
    dynetExpression_t **newobj);

/**
 * Computes batched hinge loss.
 * @param x A mini-batch of vectors of scores.
 * @param indices The indices of the correct candidates for each batch element.
 * @param n Number of indices.
 * @param m The margin.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyHinge(
    const dynetExpression_t *x, const uint32_t *indices, size_t n, float m,
    dynetExpression_t **newobj);

/**
 * Computes dimensionwise hinge loss.
 * @param x A matrix of scores.
 * @param indices The indices of the correct candidate (equal in length to the
 *                dimension not specified by "d").
 * @param n Number of indices.
 * @param d The dimension over which to calculate the loss (0 or 1).
 * @param m The margin.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyHingeDimOne(
    const dynetExpression_t *x, const uint32_t *indices, size_t n, uint32_t d,
    float m, dynetExpression_t **newobj);

/**
 * Computes batched dimensionwise hinge loss.
 * @param x A mini-batch of matrices of scores.
 * @param indices The indices of the correct candidates for each batch element.
 * @param n Number of indices.
 * @param d The dimension over which to calculate the loss (0 or 1).
 * @param m The margin.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyHingeDim(
    const dynetExpression_t *x, const uint32_t *indices, size_t n, uint32_t d,
    float m, dynetExpression_t **newobj);

/**
 * Computes sparsemax.
 * @param x A vector of scores.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySparsemax(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes sparsemax loss.
 * @param x A vector of scores.
 * @param target_support The target correct labels.
 * @param n Number of labels.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySparsemaxLoss(
    const dynetExpression_t *x, const uint32_t *target_support, size_t n,
    dynetExpression_t **newobj);

/**
 * Computes constrained softmax.
 * @param x A vector of scores.
 * @param y A vector of upper bound constraints on probabilities.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstrainedSoftmax(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes squared norm.
 * @param x A vector of values.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySquaredNorm(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes L2 norm.
 * @param x A vector of values.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyL2Norm(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes squared distance.
 * @param x A vector of values.
 * @param y Another vector of values.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySquaredDistance(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes L1 distance.
 * @param x A vector of values.
 * @param y Another vector of values.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyL1Distance(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes huber distance.
 * @param x A vector of values.
 * @param y Another vector of values.
 * @param c The parameter of the huber distance parameterizing the cuttoff.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyHuberDistance(
    const dynetExpression_t *x, const dynetExpression_t *y, float c,
    dynetExpression_t **newobj);

/**
 * Computes binary log loss.
 * @param x A vector of values.
 * @param y A vector of true answers.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyBinaryLogLoss(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Computes pairwise rank loss.
 * @param x A vector of values.
 * @param y A vector of true answers.
 * @param m The margin.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPairwiseRankLoss(
    const dynetExpression_t *x, const dynetExpression_t *y, float m,
    dynetExpression_t **newobj);

/**
 * Computes Poisson loss.
 * @param x The parameter of the Poisson distribution.
 * @param y The target value.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPoissonLoss(
    const dynetExpression_t *x, uint32_t y, dynetExpression_t **newobj);

/**
 * Prevents backprop.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyNobackprop(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Flips gradient.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyFlipGradient(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Scales gradient by constant.
 * @param x Input expression.
 * @param lambd scale.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyScaleGradient(
    const dynetExpression_t *x, float lambd, dynetExpression_t **newobj);

/**
 * Computes argmax.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyArgmaxWithZeroGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Computes argmax.
 * @param x Input expression.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyArgmaxWithStraightThroughGradientMode(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Reshapes to another size.
 * @param x Input expression.
 * @param d New dimension.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyReshape(
    const dynetExpression_t *x, const dynetDim_t *d,
    dynetExpression_t **newobj);

/**
 * Transposes a matrix.
 * @param x Input expression.
 * @param dims The dimensions to swap.
 * @param n Number of specified dimensions.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyTranspose(
    const dynetExpression_t *x, const uint32_t *dims, size_t n,
    dynetExpression_t **newobj);

/**
 * Selects rows.
 * @param x Input expression.
 * @param rows The rows to extract.
 * @param n Number of specified rows.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySelectRows(
    const dynetExpression_t *x, const uint32_t *rows, size_t n,
    dynetExpression_t **newobj);

/**
 * Selects columns.
 * @param x Input expression.
 * @param cols The columns to extract.
 * @param n Number of specified columns.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplySelectCols(
    const dynetExpression_t *x, const uint32_t *cols, size_t n,
    dynetExpression_t **newobj);

/**
 * Picks element.
 * @param x Input expression.
 * @param v The index of the element to select.
 * @param d The dimension along which to choose the element.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPickOne(
    const dynetExpression_t *x, uint32_t v, uint32_t d,
    dynetExpression_t **newobj);

/**
 * Picks elements from batches.
 * @param x Input expression.
 * @param v A vector of indicies to choose, one for each batch in the input
 *          expression.
 * @param n Number of indices.
 * @param d The dimension along which to choose the elements.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPick(
    const dynetExpression_t *x, const uint32_t *v, size_t n, uint32_t d,
    dynetExpression_t **newobj);

/**
 * Picks range of elements.
 * @param x Input expression.
 * @param s The start index.
 * @param e The end index.
 * @param d The dimension along which to pick.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPickRange(
    const dynetExpression_t *x, uint32_t s, uint32_t e, uint32_t d,
    dynetExpression_t **newobj);

/**
 * Picks batch element.
 * @param x Input expression.
 * @param v The index of the batch element to be picked.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPickBatchElem(
    const dynetExpression_t *x, uint32_t v, dynetExpression_t **newobj);

/**
 * Picks batch elements.
 * @param x Input expression.
 * @param v A vector of indicies of the batch elements to be picked.
 * @param n Number of indices.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyPickBatchElems(
    const dynetExpression_t *x, const uint32_t *v, size_t n,
    dynetExpression_t **newobj);

/**
 * Stridingly selects in multiple dimensions.
 * @param x Input expression.
 * @param strides List of strides for each dimension, must be >= 1. Dimensions
 *                not included default to 1. Batch dimension can be included as
 *                very last dimension.
 * @param n_strides Number of strides.
 * @param from List of 0-based offsets (inclusive) for each dimension, must be
 *             >= 0. Dimensions not included default to 0. Batch dimension can
 *             be included as very last dimension.
 * @param n_from Number of `from` offsets.
 * @param to List of highest 0-based index to select (exclusive) for each
 *           dimension, must be >= 0. Dimensions not included default to the
 *           corresponding dim size. Batch dimension can be included as very
 *           last dimension.
 * @param n_to Number of `to` offsets.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyStridedSelect(
    const dynetExpression_t *x, const int32_t *strides, size_t n_strides,
    const int32_t *from, size_t n_from, const int32_t *to, size_t n_to,
    dynetExpression_t **newobj);

/**
 * Concatenates list of expressions to a single batched expression.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConcatenateToBatch(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Concatenates columns.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConcatenateCols(
    const dynetExpression_t *const *xs, size_t n, dynetExpression_t **newobj);

/**
 * Concatenates expressions.
 * @param xs Input expressions.
 * @param n Number of inputs.
 * @param d The dimension along which to perform concatenation.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConcatenate(
    const dynetExpression_t *const *xs, size_t n, uint32_t d,
    dynetExpression_t **newobj);

/**
 * Selects max out through a dimension.
 * @param x Input expression.
 * @param d The dimension along which to choose the element.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMaxDim(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj);

/**
 * Selects min out through a dimension.
 * @param x Input expression.
 * @param d The dimension along which to choose the element.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMinDim(
    const dynetExpression_t *x, uint32_t d, dynetExpression_t **newobj);

/**
 * Adds Gaussian noise.
 * @param x Input expression.
 * @param stddev The standard deviation of the Gaussian.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyNoise(
    const dynetExpression_t *x, float stddev, dynetExpression_t **newobj);

/**
 * Applies dropout.
 * @param x Input expression.
 * @param p The dropout probability.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyDropout(
    const dynetExpression_t *x, float p, dynetExpression_t **newobj);

/**
 * Applies dropout along a specific dimension.
 * @param x Input expression.
 * @param d The dimension along which to drop.
 * @param p The dropout probability.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyDropoutDim(
    const dynetExpression_t *x, uint32_t d, float p,
    dynetExpression_t **newobj);

/**
 * Applies dropout to entire elements of a minibatch.
 * @param x Input expression.
 * @param p The dropout probability.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyDropoutBatch(
    const dynetExpression_t *x, float p, dynetExpression_t **newobj);

/**
 * Applies block dropout.
 * @param x Input expression.
 * @param p The block dropout probability.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyBlockDropout(
    const dynetExpression_t *x, float p, dynetExpression_t **newobj);

DYNET_C_API DYNET_C_STATUS dynetApplyFilter1dNarrow(
    const dynetExpression_t *x, const dynetExpression_t *f,
    dynetExpression_t **newobj);

/**
 * Selects out k maximum values along a given dimension.
 * @param x Input expression.
 * @param k Number of maximum values to retrieve along the given dimension.
 * @param d Dimension on which to perform kmax-pooling.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyKmaxPooling(
    const dynetExpression_t *x, uint32_t k, uint32_t d,
    dynetExpression_t **newobj);

DYNET_C_API DYNET_C_STATUS dynetApplyFoldRows(
    const dynetExpression_t *x, uint32_t nrows,
    dynetExpression_t **newobj);

DYNET_C_API DYNET_C_STATUS dynetApplyAverageCols(
    const dynetExpression_t *x, dynetExpression_t **newobj);

DYNET_C_API DYNET_C_STATUS dynetApplyKmhNgram(
    const dynetExpression_t *x, uint32_t n, dynetExpression_t **newobj);

/**
 * Applies 2D convolution operation without bias parameters.
 * @param x The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with
 *          an optional batch dimension.
 * @param f 2D convolution filters: H x W x Ci x Co (ColMaj), 4D tensor.
 * @param stride The row and column strides.
 * @param n The number of strides.
 * @param is_valid 'VALID' convolution or 'SAME' convolution.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConv2d(
    const dynetExpression_t *x, const dynetExpression_t *f,
    const uint32_t *stride, size_t n, DYNET_C_BOOL is_valid,
    dynetExpression_t **newobj);

/**
 * Applies 2D convolution operation with bias parameters.
 * @param x The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with
 *          an optional batch dimension.
 * @param f 2D convolution filters: H x W x Ci x Co (ColMaj), 4D tensor.
 * @param b The bias (1D: Ci).
 * @param stride The row and column strides.
 * @param n The number of strides.
 * @param is_valid 'VALID' convolution or 'SAME' convolution.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConv2dWithBias(
    const dynetExpression_t *x, const dynetExpression_t *f,
    const dynetExpression_t *b, const uint32_t *stride, size_t n,
    DYNET_C_BOOL is_valid, dynetExpression_t **newobj);

/**
 * Applies 2D maxpooling operation.
 * @param x The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with
 *          an optional batch dimension.
 * @param ksize The height and width of the maxpooling2d window or kernel.
 * @param n_ksize The number of ksize inputs.
 * @param stride The row and column strides
 * @param n_stride The number of strides.
 * @param is_valid 'VALID' or 'SAME'.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyMaxpooling2d(
    const dynetExpression_t *x, const uint32_t *ksize, size_t n_ksize,
    const uint32_t *stride, size_t n_stride, DYNET_C_BOOL is_valid,
    dynetExpression_t **newobj);

/**
 * Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor.
 * @param x Rank 3 tensor.
 * @param y Vector.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyContract3d1d(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor with an
 * additional bias parameter.
 * @param x Rank 3 tensor.
 * @param y Vector.
 * @param b Bias matrix.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyContract3d1dWithBias(
    const dynetExpression_t *x, const dynetExpression_t *y,
    const dynetExpression_t *b, dynetExpression_t **newobj);

/**
 * Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor.
 * @param x Rank 3 tensor.
 * @param y Vector.
 * @param z Vector.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyContract3d1d1d(
    const dynetExpression_t *x, const dynetExpression_t *y,
    const dynetExpression_t *z, dynetExpression_t **newobj);

/**
 * Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor with an
 * additional bias parameter.
 * @param x Rank 3 tensor.
 * @param y Vector.
 * @param z Vector.
 * @param b Bias vector.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyContract3d1d1dWithBias(
    const dynetExpression_t *x, const dynetExpression_t *y,
    const dynetExpression_t *z, const dynetExpression_t *b,
    dynetExpression_t **newobj);

/**
 * Takes the inverse of a matrix.
 * @param x A square matrix.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyInverse(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Takes the log of the determinant of a matrix.
 * @param x A square matrix.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLogdet(
    const dynetExpression_t *x, dynetExpression_t **newobj);

/**
 * Takes the trace of the product of matrices.
 * @param x A matrix.
 * @param y Another matrix.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyTraceOfProduct(
    const dynetExpression_t *x, const dynetExpression_t *y,
    dynetExpression_t **newobj);

/**
 * Performs layer normalization.
 * @param x Input expression (possibly batched).
 * @param g Gain (same dimension as x, no batch dimension).
 * @param b Bias (same dimension as x, no batch dimension).
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyLayerNorm(
    const dynetExpression_t *x, const dynetExpression_t *g,
    const dynetExpression_t *b, dynetExpression_t **newobj);

/**
 * Performs weight normalization.
 * @param w Input expression (weight parameter).
 * @param g Gain (scalar expression, usually also a parameter).
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyWeightNorm(
    const dynetExpression_t *w, const dynetExpression_t *g,
    dynetExpression_t **newobj);

/**
 * Copies tensor between devices.
 * @param x Input expression.
 * @param device Device to place return tensor.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyToDevice(
    const dynetExpression_t *x, dynetDevice_t *device,
    dynetExpression_t **newobj);

#endif  // DYNET_C_EXPR_H_
