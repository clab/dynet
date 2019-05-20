#ifndef DYNET_C_IO_H_
#define DYNET_C_IO_H_

#include <dynet_c/define.h>
#include <dynet_c/model.h>

/**
 * Opaque type of TextFileSaver.
 */
typedef struct dynetTextFileSaver dynetTextFileSaver_t;

/**
 * Opaque type of TextFileLoader.
 */
typedef struct dynetTextFileLoader dynetTextFileLoader_t;

/**
 * Creates a new TextFileSaver object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateTextFileSaver(
    const char *filename, DYNET_C_BOOL append, dynetTextFileSaver_t **newobj);

/**
 * Deletes the TextFileSaver object.
 * @param saver Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteTextFileSaver(
    dynetTextFileSaver_t *saver);

/**
 * Saves a ParameterCollection using the TextFileSaver object.
 * @param saver Pointer of a handler.
 * @param model ParameterCollection object to be saved.
 * @param key Name for the ParameterCollection in the saved file.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSaveParameterCollection(
    dynetTextFileSaver_t *saver, const dynetParameterCollection_t *model,
    const char *key);

/**
 * Saves a Parameter using the TextFileSaver object.
 * @param saver Pointer of a handler.
 * @param param Parameter object to be saved.
 * @param key Name for the Parameter.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSaveParameter(
    dynetTextFileSaver_t *saver, const dynetParameter_t *param,
    const char *key);

/**
 * Saves a LookupParameter using the TextFileSaver object.
 * @param saver Pointer of a handler.
 * @param param LookupParameter object to be saved.
 * @param key Name for the LookupParameter.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetSaveLookupParameter(
    dynetTextFileSaver_t *saver, const dynetLookupParameter_t *param,
    const char *key);

/**
 * Creates a new TextFileLoader object.
 * @param newobj Pointer to receive a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetCreateTextFileLoader(
    const char *filename, dynetTextFileLoader_t **newobj);

/**
 * Deletes the TextFileLoader object.
 * @param loader Pointer of a handler.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetDeleteTextFileLoader(
    dynetTextFileLoader_t *loader);

/**
 * Loads a ParameterCollection using the TextFileLoader object.
 * @param loader Pointer of a handler.
 * @param model ParameterCollection object to be populated.
 * @param key Name corresponding to the ParameterCollection.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetPopulateParameterCollection(
    dynetTextFileLoader_t *loader, dynetParameterCollection_t *model,
    const char *key);

/**
 * Loads a Parameter using the TextFileLoader object.
 * @param loader Pointer of a handler.
 * @param param Parameter object to be populated in.
 * @param key Name for the Parameter.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetPopulateParameter(
    dynetTextFileLoader_t *loader, dynetParameter_t *param, const char *key);

/**
 * Loads a LookupParameter using the TextFileLoader object.
 * @param loader Pointer of a handler.
 * @param param LookupParameter object to be populated in.
 * @param key Name for the LookupParameter.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetPopulateLookupParameter(
    dynetTextFileLoader_t *loader, dynetLookupParameter_t *param,
    const char *key);

/**
 * Loads a Parameter from the ParameterCollection using the
 * TextFileLoader object.
 * @param loader Pointer of a handler.
 * @param model ParameterCollection object to load a lookup parameter.
 * @param key Name for the Parameter.
 * @param param Pointer to receive the Parameter object.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetLoadParameterFromParameterCollection(
    dynetTextFileLoader_t *loader, dynetParameterCollection_t *model,
    const char *key, dynetParameter_t **param);

/**
 * Loads a LookupParameter from the ParameterCollection using the
 * TextFileLoader object.
 * @param loader Pointer of a handler.
 * @param model ParameterCollection object to load a lookup parameter.
 * @param key Name for the LookupParameter.
 * @param param Pointer to receive the LookupParameter object.
 * @return Status code.
 */
DYNET_C_API DYNET_C_STATUS dynetLoadLookupParameterFromParameterCollection(
    dynetTextFileLoader_t *loader, dynetParameterCollection_t *model,
    const char *key, dynetLookupParameter_t **param);

#endif  // DYNET_C_IO_H_
