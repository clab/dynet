#include "dynet/io.h"

namespace dynet {

void Pack::save(ParameterCollection & model,
                const std::string & key, bool is_append) {
  std::string key_str(key);
  if (key.size() == 0) {
    key_str = model.get_namespace();
  }
  if (duplicate_key_check(key_str) == false) {
    DYNET_RUNTIME_ERR("You couldn't save ParameterCollections with the same key in file: " + fn);
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

void Pack::save(ParameterCollection & model,
                     const std::vector<std::string> & filter_lst,
                     const std::string & key, bool is_append) {
  // TODO
}

void Pack::load(ParameterCollection & model, const std::string & key) {
  this->deserialize(model, key);
}

void Pack::load(ParameterCollection & model,
                const std::vector<std::string> & filter_lst,
                const std::string & key) {
  // TODO
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

void Pack::serialize(ParameterCollection & model, const std::string & key, bool is_append) {
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

  // read parameters
  std::getline(f, line);
  while (line == "#Parameter#") {
    std::getline(f, line);
    auto name = dynet::str_split(line, '/').back();
    name = name.substr(0, name.find_first_of("__"));

    Dim d;
    std::getline(f, line);
    std::istringstream iss(line);
    iss >> d;

    // add param into input model
    Parameter param = model.add_parameters(d, name);

    // read param.get_storage().values
    std::vector<float> params_lst;
    auto deserialize_tensor_lambda = [&] () {
      for (int k1 = 0; k1 < d.d[0]; ++k1) {
        // TODO: check dimensions 
        int sz = d.nd == 1 ? 1 : d.d[1];
        std::vector<float> tmp(sz);
        std::getline(f, line);
        std::istringstream iss(line);
        iss >> tmp;
        params_lst.insert(params_lst.end(), tmp.begin(), tmp.end());
      }
    };
    deserialize_tensor_lambda();
    std::vector<float> params_order_lst = params_lst;
    auto transpose_lambda = [&] () {
      for (size_t k = 0; k < params_lst.size(); ++k) {
        int i = k / d.d[1], j = k % d.d[1];
        int indx = j * d.d[0] + i;
        params_order_lst[indx] = params_lst[k];
      }
      params_lst.resize(0);
    };
    // TODO: high dimensions >=3
    if (d.nd == 2) {
      transpose_lambda();
    }
    TensorTools::SetElements(param.get_storage().values, params_order_lst);

    // read param.get_storage().g
    params_lst.resize(0);
    deserialize_tensor_lambda();
    params_order_lst = params_lst;
    if (d.nd == 2) {
      transpose_lambda();
    }
    TensorTools::SetElements(param.get_storage().g, params_order_lst);
    std::getline(f, line);
  } // while Parameter

  // read lookup parameters
  while (line == "#LookupParameter#") {
    std::getline(f, line);
    auto name = dynet::str_split(line, '/').back();
    name = name.substr(0, name.find_first_of("__"));

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
    LookupParameter lookup_param = model.add_lookup_parameters(N, d, name);

    // read lookup_param.get_storage().all_values
    std::vector<float> lookup_params_lst;
    auto deserialize_tensor_lambda = [&] () {
      for (int k1 = 0; k1 < all_dim.d[0]; ++k1) {
        // TODO: check dimensions 
        std::vector<float> tmp(all_dim.d[1]);
        std::getline(f, line);
        std::istringstream iss(line);
        iss >> tmp;
        lookup_params_lst.insert(lookup_params_lst.end(), tmp.begin(), tmp.end());
      }
    };
    deserialize_tensor_lambda();
    std::vector<float> lookup_params_order_lst = lookup_params_lst;
    auto transpose_lambda = [&] () {
      for (size_t k = 0; k < lookup_params_lst.size(); ++k) {
        int i = k / all_dim.d[1], j = k % all_dim.d[1];
        int indx = j * all_dim.d[0] + i;
        lookup_params_order_lst[indx] = lookup_params_lst[k];
      }
      lookup_params_lst.resize(0);
    };
    if (all_dim.nd == 2) {
      transpose_lambda();
    }
    TensorTools::SetElements(lookup_param.get_storage().all_values,
                             lookup_params_order_lst);

    // read lookup_param.get_storage().all_grads
    lookup_params_lst.resize(0);
    deserialize_tensor_lambda();
    lookup_params_order_lst = lookup_params_lst;
    // TODO: high dimensions >=3
    if (all_dim.nd == 2) {
      transpose_lambda();
    }
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

} // namespace dynet
