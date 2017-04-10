#include "dynet/io.h"

namespace dynet {

void Pack::save(const ParameterCollection & model,
                const std::string & key, bool is_append) {
  std::string key_str(key);
  if (key.size() == 0) {
    key_str = model.get_namespace();
  }
  if (duplicate_key_check(key_str) == false) {
    DYNET_RUNTIME_ERR("You couldn't save ParameterCollections with the same key " + key_str + " in file: " + fn);
  }
  // write offset info into meta file
  std::ofstream os;
  if (is_append) {
    os.open(fn_meta, std::ofstream::app);
  } else {
    os.open(fn_meta);
  }
  os << key_str << ':' << offset << '\n';
  os.close();
  // write model into model file
  this->serialize(model, key, is_append);
}

void Pack::save(const ParameterCollection & model,
                const std::vector<std::string> & filter_lst,
                const std::string & key, bool is_append) {
  DYNET_RUNTIME_ERR("This interface is not implemented yet for Pack object.");
}

void Pack::populate(ParameterCollection & model, const std::string & key) {
  this->deserialize(model, key);
}

void Pack::populate(ParameterCollection & model,
                    const std::vector<std::string> & filter_lst,
                    const std::string & key) {
  DYNET_RUNTIME_ERR("This interface is not implemented yet for Pack object.");
}

bool Pack::duplicate_key_check(const std::string & key) {
  std::ifstream f(fn_meta);
  std::string line;
  while (std::getline(f, line)) {
    auto kv = dynet::str_split(line, ':');
    if (kv[0] == key) return false;
  }
  f.close();
  return true;
}

void Pack::serialize(const ParameterCollection & model, const std::string & key, bool is_append) {
  std::ofstream os;
  if (is_append) {
    os.open(fn, std::ofstream::app);
  } else {
    os.open(fn);
  }
  os.seekp(this->offset);
  os << '#' << std::endl; // identifier of beginning of the ParameterCollection
  auto params = model.get_parameter_storages();
  for (auto & param : params) {
    os << "#Parameter#" << std::endl;
    os << param->name << std::endl;
    os << param->dim << std::endl;
    os << param->values << std::endl;
    os << param->g << std::endl;
  }
  auto lookup_params = model.get_lookup_parameter_storages();
  for (auto & lookup_param: lookup_params) {
    os << "#LookupParameter#" << std::endl;
    os << lookup_param->name << std::endl;
    os << lookup_param->all_dim << std::endl;
    os << lookup_param->dim << std::endl;
    os << lookup_param->all_values << std::endl;
    os << lookup_param->all_grads << std::endl;
  }
  this->offset = os.tellp();
  os.close();
}

void Pack::deserialize(ParameterCollection & model, const std::string & key) {
  std::ifstream meta_f(fn_meta);
  std::ifstream f(fn);
  // find the offset of the key
  long long local_offset = -1;

  std::string line;
  if (key.size() == 0) {
    // case for no key specified
    local_offset = 0;
  } else {
    while (std::getline(meta_f, line)) {
      auto kv = dynet::str_split(line, ':');
      if (kv[0] == key) {
        local_offset = std::stoll(kv[1]);
        break;
      }
    }
  }
  if (local_offset == -1) {
    DYNET_RUNTIME_ERR("Load error: no such key: " + key);
  }

  // check identifier
  f.seekg(local_offset);
  std::getline(f, line);
  if (line != "#") {
    DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
  }

  std::string ns = model.get_namespace();
  // read parameters
  std::getline(f, line);
  while (line == "#Parameter#") {
    std::getline(f, line);
    if (dynet::startswith(line, ns) == false) {
      DYNET_RUNTIME_ERR("Inconsistent namespace error: " + line + " | " + ns);
    }
    auto name = line;

    Dim d;
    std::getline(f, line);
    std::istringstream iss(line);
    iss >> d;

    // add param into input model
    Parameter param = model.add_parameters(d);
    param.get_storage().name = name;

    // read param.get_storage().values
    std::vector<float> params_order_lst;
    deserialize_tensor(f, d, params_order_lst);
    TensorTools::SetElements(param.get_storage().values, params_order_lst);

    // read param.get_storage().g
    params_order_lst.resize(0);
    deserialize_tensor(f, d, params_order_lst);
    TensorTools::SetElements(param.get_storage().g, params_order_lst);
    std::getline(f, line);
  } // while Parameter

  // read lookup parameters
  while (line == "#LookupParameter#") {
    std::getline(f, line);
    if (dynet::startswith(line, ns) == false) {
      DYNET_RUNTIME_ERR("Inconsistent namespace error: " + line + " | " + ns);
    }
    auto name = line;

    Dim all_dim;
    std::getline(f, line);
    std::istringstream iss(line);
    iss >> all_dim;
    unsigned int N = all_dim.d[all_dim.nd - 1];

    Dim d;
    std::getline(f, line);
    std::istringstream iss2(line);
    iss2 >> d;

    // add lookup_param into input model
    LookupParameter lookup_param = model.add_lookup_parameters(N, d);
    lookup_param.get_storage().name = name;

    // read lookup_param.get_storage().all_values
    std::vector<float> lookup_params_order_lst;
    deserialize_tensor(f, all_dim, lookup_params_order_lst);
    TensorTools::SetElements(lookup_param.get_storage().all_values,
                             lookup_params_order_lst);

    // read lookup_param.get_storage().all_grads
    lookup_params_order_lst.resize(0);
    deserialize_tensor(f, all_dim, lookup_params_order_lst);
    TensorTools::SetElements(lookup_param.get_storage().all_grads,
                             lookup_params_order_lst);
    std::getline(f, line);
  } // while LookupParameter

  if (line.size()) {
    if (line != "#") {
      DYNET_RUNTIME_ERR("Invalid model file format. Check this line: " + line);
    }
  }
  f.close();
  meta_f.close();
}
  
void Pack::deserialize_tensor(std::ifstream & f, const Dim & d, std::vector<float> & params_order_lst) {
  std::string line;
  std::vector<float> params_lst;
  for (int k1 = 0; k1 < d.d[0]; ++k1) {
    // TODO: dimension check
    int sz = d.nd == 1 ? 1 : d.d[1];
    std::vector<float> tmp(sz);
    std::getline(f, line);
    std::istringstream iss(line);
    iss >> tmp;
    params_lst.insert(params_lst.end(), tmp.begin(), tmp.end());
  }
  params_order_lst = params_lst;
  if (d.nd == 2) {
    // transpose
    // TODO: dimension >= 3
    for (size_t k = 0; k < params_lst.size(); ++k) {
      int i = k / d.d[1], j = k % d.d[1];
      int indx = j * d.d[0] + i;
      params_order_lst[indx] = params_lst[k];
    }
  }
}

} // namespace dynet
