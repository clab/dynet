#include "dynet/io.h"
#include "dynet/tensor.h"

namespace dynet {

TextFileSaver::TextFileSaver(const std::string & filename, bool append) :
        datastream(filename, append ? std::ofstream::app : std::ofstream::out),
        metastream(filename + ".meta", append ? std::ofstream::app : std::ofstream::out) {
  if(!datastream)
    DYNET_RUNTIME_ERR("Could not write model to " << filename);
  if(!metastream)
    DYNET_RUNTIME_ERR("Could not write model metadata to " << (filename + ".meta"));
}

void TextFileSaver::save(const ParameterCollection & model,
                          const std::string & key) {
  std::string key_str(key);
  if (key.size() == 0)
    key_str = model.get_namespace();
  // if (duplicate_key_check(key_str) == false)
  //   DYNET_RUNTIME_ERR("You couldn't save ParameterCollections with the same key " + key_str + " in file: " + fn);
  std::unordered_map<std::string, long long> offset_dict;
  metastream << key_str << ':' << offset;
  this->serialize(model, key, offset_dict);
  for (auto & kv : offset_dict)
    metastream << '|' << kv.first << ':' << kv.second;
  metastream << '\n';
}

void TextFileSaver::save(const ParameterCollection & model,
                const std::vector<std::string> & filter_lst,
                const std::string & key) {
  DYNET_RUNTIME_ERR("This interface is not implemented yet for TextFileSaver object.");
}

void TextFileSaver::save(const Parameter & param, const std::string & key) {
  std::string key_str(key);
  if (key.size() == 0)
    key_str = param.get_fullname();
  // if (duplicate_key_check(key_str) == false)
  //   DYNET_RUNTIME_ERR("You couldn't save Parameter with the same key " + key_str + " in file: " + fn);
  metastream << key_str << ':' << offset << '\n';
  this->serialize(param, key);
}

void TextFileSaver::save(const LookupParameter & lookup_param, const std::string & key) {
  std::string key_str(key);
  if (key.size() == 0)
    key_str = lookup_param.get_fullname();
  // if (duplicate_key_check(key_str) == false)
  //   DYNET_RUNTIME_ERR("You couldn't save LookupParameter with the same key " + key_str + " in file: " + fn);
  metastream << key_str << ':' << offset << '\n';
  this->serialize(lookup_param, key);
}

void TextFileSaver::serialize_parameter(std::ofstream & os, const ParameterStorage *p) {
  os << p->name << '\n' << p->dim << '\n';
  os << dynet::as_vector(p->values);
  os << dynet::as_vector(p->g);
}

void TextFileSaver::serialize_lookup_parameter(std::ofstream & os,
                                      const LookupParameterStorage *p) {
  os << p->name << '\n' << p->all_dim << '\n' << p->dim << '\n';
  os << dynet::as_vector(p->all_values);
  os << dynet::as_vector(p->all_grads);
}

void TextFileLoader::populate(ParameterCollection & model, const std::string & key) {
  this->deserialize(model, key);
}

void TextFileLoader::populate(ParameterCollection & model,
                    const std::vector<std::string> & filter_lst,
                    const std::string & key) {
  DYNET_RUNTIME_ERR("This interface is not implemented yet for TextFileLoader object.");
}

void TextFileLoader::populate(Parameter & param,
                    const std::string & key) {
  this->deserialize(param, key);
}

void TextFileLoader::populate(LookupParameter & lookup_param,
                    const std::string & key) {
  this->deserialize(lookup_param, key);
}

Parameter TextFileLoader::load_param(ParameterCollection & model,
                           const std::string & key) {
  return this->deserialize_param(model, key);
}

LookupParameter TextFileLoader::load_lookup_param(ParameterCollection & model,
                                        const std::string & key) {
  return this->deserialize_lookup_param(model, key);
}

void TextFileSaver::serialize(const ParameterCollection & model,
                     const std::string & key,
                     std::unordered_map<std::string, long long> & offset_dict) {
  datastream << '#' << std::endl; // beginning identifier
  auto all_params = model.get_parameter_storages_base();
  auto params = model.get_parameter_storages();
  auto lookup_params = model.get_lookup_parameter_storages();
  size_t i = 0, j = 0;
  for (size_t k = 0; k < all_params.size();  ++k) {
    if (i < params.size() && all_params[k] == params[i]) {
      datastream << "#Parameter#" << std::endl;
      offset_dict[params[i]->name] = datastream.tellp();
      serialize_parameter(datastream, params[i]);
      ++i;
    } else {
      datastream << "#LookupParameter#" << std::endl;
      offset_dict[lookup_params[j]->name] = datastream.tellp();
      serialize_lookup_parameter(datastream, lookup_params[j]);
      ++j;
    }
  }
  this->offset = datastream.tellp();
}

void TextFileSaver::serialize(const Parameter & param,
                     const std::string & key) {
  datastream << '#' << std::endl; // beginning identifier
  datastream << "#Parameter#" << std::endl;
  serialize_parameter(datastream, param.p);
  this->offset = datastream.tellp();
}

void TextFileSaver::serialize(const LookupParameter & lookup_param,
                              const std::string & key) {
  datastream << '#' << std::endl;
  datastream << "#LookupParameter#" << std::endl;
  serialize_lookup_parameter(datastream, lookup_param.p);
  this->offset = datastream.tellp();
}

#define TFL_OPEN_FILE() \
  std::ifstream datastream(dataname); \
  std::ifstream metastream(metaname); \
  if(!datastream) DYNET_RUNTIME_ERR("Could not read model from " << dataname); \
  if(!metastream) DYNET_RUNTIME_ERR("Could not read model metadata from " << metaname);

TextFileLoader::TextFileLoader(const std::string & filename) :
        dataname(filename), metaname(filename + ".meta") { }

void TextFileLoader::deserialize(ParameterCollection & model, const std::string & key) {
  TFL_OPEN_FILE();
  // find the offset of the key
  long long local_offset = -1;

  std::string line;
  if (key.size() == 0) {
    // case for no key specified
    local_offset = 0;
  } else {
    while (std::getline(metastream, line)) {
      auto tmp_str = dynet::str_split(line, '|').front();
      auto kv = dynet::str_split(tmp_str, ':');
      if (kv[0] == key) {
        local_offset = std::stoll(kv[1]);
        break;
      }
    }
  }
  if (local_offset == -1) {
    DYNET_RUNTIME_ERR("Load error: no such key: " + key);
  }

  datastream.seekg(local_offset);
  std::getline(datastream, line);
  if (line != "#") {
    DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
  }

  std::string ns = model.get_namespace();
  // read parameters
  std::getline(datastream, line);
  while (line == "#Parameter#" || line == "#LookupParameter#") {
    if (line == "#Parameter#") {
      std::getline(datastream, line);
      if (dynet::startswith(line, ns) == false) {
        DYNET_RUNTIME_ERR("Inconsistent namespace error: " + line + " | " + ns);
      }
      auto name = line;

      Dim d;
      std::getline(datastream, line);
      std::istringstream iss(line); iss >> d;

      // add param into input model
      Parameter param = model.add_parameters(d);
      param.get_storage().name = name;

      // read param.get_storage().values
      std::getline(datastream, line);
      std::vector<real> params_l(d.size());
      std::istringstream iss2(line); iss2 >> params_l;
      TensorTools::SetElements(param.get_storage().values, params_l);

      // read param.get_storage().g
      std::getline(datastream, line);
      std::istringstream iss3(line); iss3 >> params_l;
      TensorTools::SetElements(param.get_storage().g, params_l);
      std::getline(datastream, line);
    } else {
      std::getline(datastream, line);
      if (dynet::startswith(line, ns) == false) {
        DYNET_RUNTIME_ERR("Inconsistent namespace error: " + line + " | " + ns);
      }
      auto name = line;

      Dim all_dim;
      std::getline(datastream, line);
      std::istringstream iss(line); iss >> all_dim;
      unsigned int N = all_dim.d[all_dim.nd - 1];

      Dim d;
      std::getline(datastream, line);
      std::istringstream iss2(line); iss2 >> d;

      // add lookup_param into input model
      LookupParameter lookup_param = model.add_lookup_parameters(N, d);
      lookup_param.get_storage().name = name;

      // read lookup_param.get_storage().all_values
      std::getline(datastream, line);
      std::vector<real> lookup_params_l(all_dim.size());
      std::istringstream iss3(line); iss3 >> lookup_params_l;
      TensorTools::SetElements(lookup_param.get_storage().all_values,
                               lookup_params_l);

      // read lookup_param.get_storage().all_grads
      std::getline(datastream, line);
      std::istringstream iss4(line); iss4 >> lookup_params_l;
      TensorTools::SetElements(lookup_param.get_storage().all_grads,
                               lookup_params_l);
      std::getline(datastream, line);
    }
  } // while

  if (line.size()) {
    if (line != "#") {
      DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
    }
  }
}
  
void TextFileLoader::deserialize(Parameter & param, const std::string & key) {
  TFL_OPEN_FILE();
  std::string line;
  datastream.seekg(this->seek_offset(key));
  std::getline(datastream, line);
  if (line != "#") {
    DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
  }
  std::getline(datastream, line); // #Parameter#
  std::getline(datastream, line); auto name = line;
  Dim d;
  std::getline(datastream, line);
  std::istringstream iss(line); iss >> d;
  if (param.get_storage().dim != d) {
    DYNET_RUNTIME_ERR("Dimension is not consistent.");
  }
  param.get_storage().name = name;
  
  std::getline(datastream, line);
  std::vector<real> params_l(d.size());
  std::istringstream iss2(line); iss2 >> params_l;
  TensorTools::SetElements(param.get_storage().values, params_l);

  std::getline(datastream, line);
  std::istringstream iss3(line); iss3 >> params_l;
  TensorTools::SetElements(param.get_storage().g, params_l);
  std::getline(datastream, line);
}

void TextFileLoader::deserialize(LookupParameter & lookup_param,
                       const std::string & key) {
  TFL_OPEN_FILE();
  std::string line;
  datastream.seekg(this->seek_offset(key));
  std::getline(datastream, line);
  if (line != "#") {
    DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
  }
  std::getline(datastream, line);
  std::getline(datastream, line);
  auto name = line;
  Dim all_dim;
  std::getline(datastream, line);
  std::istringstream iss(line); iss >> all_dim;
  
  std::getline(datastream, line);
  if (lookup_param.get_storage().all_dim != all_dim) {
    DYNET_RUNTIME_ERR("Dimension is not consistent.");
  }

  lookup_param.get_storage().name = name;
  
  std::getline(datastream, line);
  std::vector<real> lookup_params_l(all_dim.size());
  std::istringstream iss2(line); iss2 >> lookup_params_l;
  TensorTools::SetElements(lookup_param.get_storage().all_values,
                           lookup_params_l);
  std::getline(datastream, line);
  std::istringstream iss3(line); iss3 >> lookup_params_l;
  TensorTools::SetElements(lookup_param.get_storage().all_grads,
                           lookup_params_l);
}

Parameter TextFileLoader::deserialize_param(ParameterCollection & model,
                                  const std::string & key) {
  TFL_OPEN_FILE();
  std::string line;
  datastream.seekg(this->seek_offset(key));
  std::getline(datastream, line);
  if (line != "#") {
    DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
  }
  std::getline(datastream, line); // #Parameter#
  std::getline(datastream, line); auto name = line;
  Dim d;
  std::getline(datastream, line);
  std::istringstream iss(line); iss >> d;
  Parameter param = model.add_parameters(d);
  param.get_storage().name = name;
  
  std::getline(datastream, line);
  std::vector<real> params_l(d.size());
  std::istringstream iss2(line); iss2 >> params_l;
  TensorTools::SetElements(param.get_storage().values, params_l);

  std::getline(datastream, line);
  std::istringstream iss3(line); iss3 >> params_l;
  TensorTools::SetElements(param.get_storage().g, params_l);
  std::getline(datastream, line);
  return param;
}

LookupParameter TextFileLoader::deserialize_lookup_param(ParameterCollection & model,
                                               const std::string & key) {
  TFL_OPEN_FILE();
  std::string line;
  datastream.seekg(this->seek_offset(key));
  std::getline(datastream, line);
  if (line != "#") {
    DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
  }
  std::getline(datastream, line);
  std::getline(datastream, line);
  auto name = line;
  Dim all_dim;
  std::getline(datastream, line);
  std::istringstream iss(line); iss >> all_dim;
  
  unsigned int N = all_dim.d[all_dim.nd - 1];
  Dim d;
  std::getline(datastream, line);
  std::istringstream iss2(line); iss2 >> d;
  LookupParameter lookup_param = model.add_lookup_parameters(N, d);

  if (lookup_param.get_storage().all_dim != all_dim) {
    DYNET_RUNTIME_ERR("Dimension is not consistent.");
  }

  lookup_param.get_storage().name = name;

  std::getline(datastream, line);
  std::vector<real> lookup_params_l(all_dim.size());
  std::istringstream iss3(line); iss3 >> lookup_params_l;
  TensorTools::SetElements(lookup_param.get_storage().all_values,
                           lookup_params_l);
  std::getline(datastream, line);
  std::istringstream iss4(line); iss4 >> lookup_params_l;
  TensorTools::SetElements(lookup_param.get_storage().all_grads,
                           lookup_params_l);
  return lookup_param;
}

long long TextFileLoader::seek_offset(const std::string & key) {
  TFL_OPEN_FILE();
  std::string line;
  long long local_offset = -1;
  if (key.size() == 0) {
    local_offset = 0;
  } else {
    while (std::getline(metastream, line)) {
      auto kv = dynet::str_split(line, ':');
      if (kv[0] == key) {
        local_offset = std::stoll(kv[1]);
        break;
      }
    }
  }
  if (local_offset== -1) {
    DYNET_RUNTIME_ERR("Load error: no such key: " + key);
  }
  return local_offset;
}

} // namespace dynet
