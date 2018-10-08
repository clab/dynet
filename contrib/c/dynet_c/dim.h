#ifndef DYNET_C_DIM_H_
#define DYNET_C_DIM_H_

#include <dynet_c/define.h>

/**
 * Opaque type of Dim.
 */
typedef struct dynetDim dynetDim_t;

/**
 * Creates a new Dim object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateDim(dynetDim_t **newobj);

/**
 * Creates a new Dim object.
 * @param newobj Pointer to receive a handler.
 * @param dims List of the dimension sizes.
 * @param n Length of the dims.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateDimWithDimensions(
    const uint32_t *dims, size_t n, dynetDim_t **newobj);

/**
 * Creates a new Dim object.
 * @param dims List of the dimension sizes.
 * @param n Length of the dims.
 * @param batch Batch size.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateDimWithDimensionsAndBatch(
    const uint32_t *dims, size_t n, uint32_t batch, dynetDim_t **newobj);

/**
 * Creates a clone of the existing Dim object.
 * @param src Pointer to a source Dim.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCloneDim(
    const dynetDim_t *src, dynetDim_t **newobj);

/**
 * Deletes the Dim object.
 * @param shape Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteDim(dynetDim_t *dim);

/**
 * Returns the total size of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the total size of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimTotalSize(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Returns the number of elements within a batch.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the product of all dimensions of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimBatchSize(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Sums all dimensions of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the sum of all dimensions of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSumDimDimensions(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Truncates trailing dimensions of 1 of the dim.
 * @param dim Pointer of a handler.
 * @param new_dim Pointer for a new dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetTruncateDim(
    const dynetDim_t *dim, dynetDim_t **new_dim);

/**
 * Changes the number of dimensions of the dim.
 * @param dim Pointer of a handler.
 * @param i New number of dimensions.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetResizeDim(dynetDim_t *dim, uint32_t i);

/**
 * Returns the number of dimensions of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the number of dimensions of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimNDimensions(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Returns the size of first dimension of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the size of first dimension of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimRows(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Returns the size of second dimension of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the size of second dimension of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimCols(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Returns the batch dimension of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the batch dimension of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimBatchElems(
    const dynetDim_t *dim, uint32_t *retval);

/**
 * Returns the size of the specific dimension of the dim.
 * @param dim Pointer of a handler.
 * @param i Index of the dimension.
 * @param retval Pointer to receive the size of the dimension of the dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetGetDimDimensionSize(
    const dynetDim_t *dim, uint32_t i, uint32_t *retval);

/**
 * Updates the size of the specific dimension of the dim.
 * @param dim Pointer of a handler.
 * @param i Index of the dimension.
 * @param s New size of the dimension.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSetDimDimensionSize(
    dynetDim_t *dim, uint32_t i, uint32_t s);

/**
 * Transposes the dim.
 * @param dim Pointer of a handler.
 * @param new_dim Pointer for a new dim.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetTransposeDim(
    const dynetDim_t *dim, dynetDim_t **new_dim);

/**
 * Compares a dim and another dim.
 * @param dim Pointer of a handler.
 * @param other Dim object to compare.
 * @param retval Pointer to receive a result: true if `dim` and `other` are
 *               same, false otherwise.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetIsDimEqualTo(
    const dynetDim_t *dim, const dynetDim_t *other, DYNET_C_BOOL *retval);

/**
 * Compares a dim and another dim.
 * @param dim Pointer of a handler.
 * @param other Dim object to compare.
 * @param retval Pointer to receive a result: true if `dim` and `other` are not
 *               same, false otherwise.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetIsNotDimEqualTo(
    const dynetDim_t *dim, const dynetDim_t *other, DYNET_C_BOOL *retval);

/**
 * Returns a string representation of the dim.
 * @param dim Pointer of a handler.
 * @param retval Pointer to receive the encoded string.
 * @param size Pointer to receive a length of the char sequence.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetRepresentDimAsString(
    const dynetDim_t *dim, char *retval, size_t *size);

#endif  // DYNET_C_DIM_H_
