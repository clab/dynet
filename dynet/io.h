#ifndef DYNET_IO_H_
#define DYNET_IO_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
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
  os << '\n';
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
  virtual ~Saver() { }

  /**
   * @brief Save ParameterCollection with key, use internal namespace if key is not given.
   *
   * @param model: input ParameterCollection object to be saved
   * @param key: optional parameter, the key for the model
   */
  virtual void save(const ParameterCollection & model,
                    const std::string & key = "") = 0;

  /**
   * @brief Save ParameterCollection's parameters and lookup parameters with filter_lst and key,
   *        use internal namespace if key is not given.
   *
   * @param model: input ParameterCollection object to be saved
   * @param filter_lst: save parameters and lookup parameters satisfy the filter_lst condition
   *                    each filter can be regex expressions
   * @param key: optional parameter, the key for the model 
   */
  virtual void save(const ParameterCollection & model,
                    const std::vector<std::string> & filter_lst,
                    const std::string & key = "") = 0;

  /**
   * @brief Save Parameter with key, use internal name if key is not given.
   *
   * @param model: input Parameter object to be saved
   * @param key: optional parameter, the key for the saving Parameter
   */
  virtual void save(const Parameter & param, const std::string & key = "") = 0;

  /**
   * @brief Save look parameter with key, use internal name if key is not given.
   * 
   * @param model: input LookupParameter object to be saved
   * @param key: optional parameter, the key for the saving Parameter
   */
  virtual void save(const LookupParameter & param, const std::string & key = "") = 0;

}; // class Saver

class Loader {
 public:
  Loader() { }
  virtual ~Loader() { }

  /**
   * @brief Populate ParameterCollection object with key.
   * 
   * @param model: input/output parameter, the ParameterCollection object to be populated in. 
   * @param key: optional parameter, the key for loading the model
   *
   */
  virtual void populate(ParameterCollection & model, const std::string & key = "") = 0;

  /**
   * @brief Populate ParameterCollection object with filter_lst and with key equals to key.
   * 
   * @param model: input/output parameter, the ParameterCollection object to be populated
   * @param filter_lst: populate parameters and lookup parameters satisfies the filter_lst condition
   *                    each filter can be regex expression
   * @param key: optional parameter, the key for loading the model
   *
   */
  virtual void populate(ParameterCollection & model,
                        const std::vector<std::string> & filter_lst,
                        const std::string & key = "") = 0;

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
  TextFileSaver(const std::string & filename, bool append = true);
  virtual ~TextFileSaver() { }
  void save(const ParameterCollection & model,
            const std::string & key = "") override;
  void save(const ParameterCollection & model,
            const std::vector<std::string> & filter_lst,
            const std::string & key = "") override;
  void save(const Parameter & param, const std::string & key = "") override;
  void save(const LookupParameter & param, const std::string & key = "") override;

protected:
  void serialize(const ParameterCollection & model,
                 const std::string & key,
                 std::unordered_map<std::string, long long> & offset_dict);
  void serialize(const Parameter & param,
                 const std::string & key);
  void serialize(const LookupParameter & lookup_param,
                 const std::string & key);
  void serialize_parameter(std::ofstream & os, const ParameterStorage *p);
  void serialize_lookup_parameter(std::ofstream & os, const LookupParameterStorage *p);

  std::ofstream datastream, metastream;
  long long offset = 0;

}; // class TextFileSaver

class TextFileLoader : public Loader {
 public:
  TextFileLoader(const std::string & filename);
  virtual ~TextFileLoader() { }
  void populate(ParameterCollection & model, const std::string & key = "") override;
  void populate(ParameterCollection & model,
                const std::vector<std::string> & filter_lst,
                const std::string & key = "") override;
  void populate(Parameter & param, const std::string & key = "") override;
  void populate(LookupParameter & lookup_param,
                const std::string & key = "") override;
  Parameter load_param(ParameterCollection & model, const std::string & key) override;
  LookupParameter load_lookup_param(ParameterCollection & model, const std::string & key) override;

 private:
  void deserialize(ParameterCollection & model, const std::string & key);
  void deserialize(Parameter & param, const std::string & key);
  void deserialize(Parameter & param,
                   const std::string & model_name,
                   const std::string & key);
  void deserialize(LookupParameter & lookup_param,
                   const std::string & key);
  void deserialize(LookupParameter & lookup_param,
                   const std::string & model_name,
                   const std::string & key);
  Parameter deserialize_param(ParameterCollection & model,
                              const std::string & key);
  Parameter deserialize_param(ParameterCollection & model,
                              const std::string & model_name,
                              const std::string & key);
  LookupParameter deserialize_lookup_param(ParameterCollection & model,
                                           const std::string & key);
  LookupParameter deserialize_lookup_param(ParameterCollection & model,
                                           const std::string & model_name,
                                           const std::string & key);
  long long seek_offset(const std::string & key);
  long long seek_offset(const std::string & model_name,
                        const std::string & key);

  std::string dataname, metaname;
  long long offset = 0;
}; // class TextFileLoader

} // namespace dynet

#endif
