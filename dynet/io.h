#ifndef DYNET_IO_H_
#define DYNET_IO_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "dynet/dim.h"
#include "dynet/model.h"
#include "dynet/tensor.h"
#include "dynet/except.h"
#include "dynet/str_util.h"

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

class Pack {
 public:
  Pack(std::string filename) : fn(filename), fn_meta(filename + ".meta") {}
  
  void reinit(std::string filename) {
    offset = 0;
    fn = filename;
    fn_meta = filename + ".meta";
  }

  /**
   * @brief Save ParameterCollection with key, use internal namespace if key is not given.
   *
   * @param model: input ParameterCollection object to be saved
   * @param key: optional parameter, the key for the model
   * @param is_append: optional parameter
   *                   to specify whether the model file is append mode or not
   */
  void save(const ParameterCollection & model,
            const std::string & key = "", bool is_append = true);

  /**
   * @brief Save ParameterCollection's parameters and lookup parameters with filter_lst and key,
   *        use internal namespace if key is not given.
   *
   * @param model: input ParameterCollection object to be saved
   * @param filter_lst: save parameters and lookup parameters satisfy the filter_lst condition
   *                    each filter can be regex expressions
   * @param key: optional parameter, the key for the model 
   * @param is_append: optional parameter
   *                   to specify whether the model file is append mode or not
   */
  void save(const ParameterCollection & model,
            const std::vector<std::string> & filter_lst,
            const std::string & key = "", bool is_append = true);

  /**
   * @brief Save Parameter with key, use internal name if key is not given.
   *
   * @param model: input Parameter object to be saved
   * @param key: optional parameter, the key for the saving Parameter
   * @param is_append: optional parameter to specify whether the model file is append mode or not
   */
  void save(const Parameter & param, const std::string & key = "", bool is_append = true);

  /**
   * @brief Save look parameter with key, use internal name if key is not given.
   * 
   * @param model: input LookupParameter object to be saved
   * @param key: optional parameter, the key for the saving Parameter
   * @param is_append: optional parameter to specify whether the model file is append mode or not
   */
  void save(const LookupParameter & param, const std::string & key = "", bool is_append = true);

  /**
   * @brief Populate ParameterCollection object with key.
   * 
   * @param model: input/output parameter, the ParameterCollection object to be populated in. 
   * @param key: optional parameter, the key for loading the model
   *
   */
  void populate(ParameterCollection & model, const std::string & key = "");

  /**
   * @brief Populate ParameterCollection object with filter_lst and with key equals to key.
   * 
   * @param model: input/output parameter, the ParameterCollection object to be populated
   * @param filter_lst: populate parameters and lookup parameters satisfies the filter_lst condition
   *                    each filter can be regex expression
   * @param key: optional parameter, the key for loading the model
   *
   */
  void populate(ParameterCollection & model,
                const std::vector<std::string> & filter_lst,
                const std::string & key = "");

  /**
   * @brief Populate independent parameter object with key.
   *        independent here means it has been saved without a ParameterCollection object
   *
   * @param param: input/output parameter, the Parameter object to be populated in. 
   * @param key: optional parameter, the key for loading the parameter 
   *
   */
  void populate(Parameter & param, const std::string & key = "");

  /**
   * @brief Populate parameter object inside a model with key.
   *
   * @param param: input/output parameter, the Parameter object to be populated in. 
   * @param model_name: model_name for holding the wanted parameter
   * @param key: the key for loading the parameter
   *
   */
  void populate(Parameter & param,
                const std::string & model_name,
                const std::string & key);

  /**
   * @brief Populate independent lookup parameter object with key.
   *        independent here means it has been saved without a LookupParameterCollection object
   *
   * @param lookup_param: input/output parameter, the LookupParameter object to be populated in. 
   * @param key: optional parameter, the key for loading the lookup parameter 
   *
   */
  void populate(LookupParameter & lookup_param,
                const std::string & key = "");

  /**
   * @brief Populate LookupParameter object inside a model with key.
   *
   * @param lookup_param: input/output parameter, the LookupParameter object to be populated in. 
   * @param model_name: model_name for holding the wanted lookup parameter
   * @param key: the key for loading the lookup parameter
   *
   */
  void populate(LookupParameter & lookup_param,
                const std::string & model_name,
                const std::string & key);
  
  /**
   * @brief Load parameter into model with key
   * 
   * @param model: input/output parameter, the model to load parameter
   * @param key: the key for loading the parameter
   * @return: the loaded parameter
   *
   */
  Parameter load_param(ParameterCollection & model, const std::string & key);

  /**
   * @brief Load parameter into model with model_name and parameter key
   * 
   * @param model: input/output parameter, the model to load parameter
   * @param model_name: model_name for holding the wanted parameter
   * @param key: the key for loading the parameter
   * @return: the loaded parameter
   *
   */
  Parameter load_param(ParameterCollection & model,
                       const std::string & model_name,
                       const std::string & key);

  /**
   * @brief Load lookup parameter into model with key
   * 
   * @param model: input/output parameter, the model to load the lookup parameter
   * @param key: the key for loading the lookup parameter
   * @return: the loaded lookup parameter
   *
   */
  LookupParameter load_lookup_param(ParameterCollection & model, const std::string & key);

  /**
   * @brief Load lookup parameter into model with model_name and lookup parameter key
   * 
   * @param model: input/output parameter, the model to load the lookup parameter
   * @param model_name: model_name for holding the wanted lookup parameter
   * @param key: the key for loading the lookup parameter
   * @return: the loaded lookup parameter
   *
   */
  LookupParameter load_lookup_param(ParameterCollection & model,
                                    const std::string & model_name,
                                    const std::string & key);

 private:
  bool duplicate_key_check(const std::string & key);
  void serialize(const ParameterCollection & model,
                 const std::string & key,
                 bool is_append,
                 std::unordered_map<std::string, long long> & offset_dict);
  void serialize(const Parameter & param,
                 const std::string & key,
                 bool is_append);
  void serialize(const LookupParameter & lookup_param,
                 const std::string & key,
                 bool is_append);
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
  void serialize_parameter(std::ofstream & os, const ParameterStorage *p);
  void serialize_lookup_parameter(std::ofstream & os, const LookupParameterStorage *p);
  long long seek_offset(const std::string & key);
  long long seek_offset(const std::string & model_name,
                        const std::string & key);

 private:
  std::string fn, fn_meta;
  long long offset = 0;
}; // class Pack

} // namespace dynet

#endif
