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

#endif  // DYNET_C_EXPR_H_
