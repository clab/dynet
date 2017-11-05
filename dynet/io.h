#ifndef DYNET_IO_H_
#define DYNET_IO_H_

#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iterator>

#include "dynet/dim.h"
#include "dynet/model.h"
#include "dynet/tensor.h"
#include "dynet/except.h"
#include "dynet/str-util.h"

namespace dynet {

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> & v) {
  for (auto & val : v) os << val << ' ';
  return os;
}

template <class T>
std::istream& operator>>(std::istream& is, std::vector<T> & v) {
  std::copy(std::istream_iterator<T>(is), std::istream_iterator<T>(), v.begin());
  return is;
}

class Saver {
 public:
  Saver() { }
  virtual ~Saver();

  /**
   * @brief Save ParameterCollection
   *
   * @param model: ParameterCollection object to be saved
   * @param key: optional parameter, the name for the ParameterCollection in the saved file. This
   *             will default to the current name of the ParameterCollection.
   * @detail: Let's say we have a ParameterCollection named "/pc1/" containing parameters
   *          "/pc1/a", "/pc1/b", and "/pc1/c". This will save the parameters with the names as-is
   *          if `key` is not specified. If `key` is specified as "/pc2/", then the parameters will
   *          be saved as "/pc2/a", "/pc2/b", and "/pc2/c".
   */
  virtual void save(const ParameterCollection & model,
                    const std::string & key = "") = 0;

  /**
   * @brief Save Parameter.
   *
   * @param model: Parameter object to be saved
   * @param key: optional parameter, the key for the parameter. This will override the Parameter's
   *             original name.
   */
  virtual void save(const Parameter & param, const std::string & key = "") = 0;

  /**
   * @brief Save look parameter with key, use internal name if key is not given.
   *
   * @param model: input LookupParameter object to be saved
   * @param key: optional parameter, the key for the parameter. This will override the Parameter's
   *             original name.
   */
  virtual void save(const LookupParameter & param, const std::string & key = "") = 0;

}; // class Saver

class Loader {
 public:
  Loader() { }
  virtual ~Loader();

  /**
   * @brief Populate the parameters of a ParameterCollection.
   *
   * @param model: The ParameterCollection to be populated.
   * @param key: optional parameter, the key corresponding to the ParameterCollection
   * @detail: This is the standard way to load parameters of a ParameterCollection from a
   *          file, and assumes that we have saved an identical ParameterCollection using
   *          Saver::save(parameter_collection).
   *          Before calling this function, we assume that the ParameterCollection has
   *          been fully specified, and all of its Parameters and LookupParameters have been
   *          created with the proper dimensions. This function will then travel through the
   *          file and load all parameters with names starting with prefix `key`, and populate
   *          the Parameters and LookupParameters one-by-one in order. When the function
   *          terminates, we must have populated all of the parameters in `model`. `key` is
   *          by default empty, so by default we will load all parameters in the file, but if
   *          we specify `key` we can load a subset of the parameters.
   *
   */
  virtual void populate(ParameterCollection & model, const std::string & key = "") = 0;

  /**
   * @brief Populate independent parameter object with key.
   *        independent here means it has been saved without a ParameterCollection object
   *
   * @param param: input/output parameter, the Parameter object to be populated in.
   * @param key: optional parameter, the key for loading the parameter
   *
   */
  virtual void populate(Parameter & param, const std::string & key = "") = 0;

  /**
   * @brief Populate independent lookup parameter object with key.
   *        independent here means it has been saved without a LookupParameterCollection object
   *
   * @param lookup_param: input/output parameter, the LookupParameter object to be populated in.
   * @param key: optional parameter, the key for loading the lookup parameter
   *
   */
  virtual void populate(LookupParameter & lookup_param,
                        const std::string & key = "") = 0;

  /**
   * @brief Load parameter into model with key
   *
   * @param model: input/output parameter, the model to load parameter
   * @param key: the key for loading the parameter
   * @return: the loaded parameter
   *
   */
  virtual Parameter load_param(ParameterCollection & model,
                               const std::string & key) = 0;

  /**
   * @brief Load lookup parameter into model with key
   *
   * @param model: input/output parameter, the model to load the lookup parameter
   * @param key: the key for loading the lookup parameter
   * @return: the loaded lookup parameter
   *
   */
  virtual LookupParameter load_lookup_param(ParameterCollection & model,
                                            const std::string & key) = 0;

}; // class Loader


class TextFileSaver : public Saver {
 public:
  TextFileSaver(const std::string & filename, bool append = false);
  ~TextFileSaver() override;
  void save(const ParameterCollection & model,
            const std::string & key = "") override;
  void save(const Parameter & param, const std::string & key = "") override;
  void save(const LookupParameter & param, const std::string & key = "") override;

protected:
  void save(const ParameterStorage & param, const std::string & key = "");
  void save(const LookupParameterStorage & param, const std::string & key = "");

  std::unique_ptr<std::ostream> p_datastream;
  std::ostream& datastream;

}; // class TextFileSaver

class TextFileLoader : public Loader {
 public:
  TextFileLoader(const std::string & filename);
  ~TextFileLoader() override;
  void populate(ParameterCollection & model, const std::string & key = "") override;
  void populate(Parameter & param, const std::string & key = "") override;
  void populate(LookupParameter & lookup_param,
                const std::string & key = "") override;
  Parameter load_param(ParameterCollection & model, const std::string & key) override;
  LookupParameter load_lookup_param(ParameterCollection & model, const std::string & key) override;

 private:
  std::string dataname;
}; // class TextFileLoader

} // namespace dynet

#endif
