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
    dynetComputationGraph_t *g, dynetParameter_t *p,
    dynetExpression_t **newobj);

/**
 * Loads constant lookup parameter.
 * @param g Computation graph.
 * @param p LookupParameter to load.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstLookupParameter(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p,
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
 * Looks up parameter.
 * @param g Computation graph.
 * @param p LookupParameter object from which to load.
 * @param index Index of the parameters within p.
 * @param newobj Pointer to receive an Expression.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetApplyConstLookupOne(
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
DYNET_C_API DYNET_C_STATUS dynetApplyConstLookup(
    dynetComputationGraph_t *g, dynetLookupParameter_t *p,
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
 * Applies sum all elements operation.
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
 * Computes log.
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
 * Computes absolute value.
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

#endif  // DYNET_C_EXPR_H_
