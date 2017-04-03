#ifndef DYNET_IO_H_
#define DYNET_IO_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "dynet/str_util.h"
#include "dynet/dim.h"
#include "dynet/tensor.h"

namespace dynet {

template <class T>
std::istream& operator>>(std::istream& is, std::vector<T> & v) {
  std::copy(std::istream_iterator<T>(is), std::istream_iterator<T>(), v.begin());
  return is;
}

class Pack {
 public:
  Pack(std::string filename) : fn(filename), fn_meta(filename + ".meta") {}

  void save(ParameterCollection & model, std::string key = "") {
    std::string key_str(key);
    if (key.size() == 0) {
      key_str = model.get_namespace();
    }

    std::ofstream os;
    os.open(fn_meta, std::ios::app);
    os << key_str << ':' << offset << '\n';
    os.close();

    this->serialize(model, key);
  }

  void load(ParameterCollection & model, std::string key = "") {
    this->deserialize(model, key);
  }
  
 private:
  void serialize(ParameterCollection & model, std::string key) {
    std::ofstream os;
    os.open(fn, std::ios::app);
    os.seekp(this->offset);
    os << '#' << std::endl;
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

  void deserialize(ParameterCollection & model, std::string key) {
    std::ifstream meta_f(fn_meta);
    std::ifstream f(fn);
    std::string key_str;
    long long local_offset = -1;
    std::string line;
    while (std::getline(meta_f, line)) {
      auto kv = dynet::str_split(line, ':');
      if (kv[0] == key) {
        local_offset = std::stoll(kv[1]);
        break;
      }
    }
    if (local_offset == -1) {
      throw std::runtime_error("Load error: no such key");
    }

    f.seekg(local_offset);
    std::getline(f, line);
    if (line != "#") {
      throw std::runtime_error("Invalid model file format.");
    }

    std::getline(f, line);
    while (line == "#Parameter#") {
      std::getline(f, line);
      auto name = dynet::str_split(line, '/').back();
      name = name.substr(0, name.find_first_of("__"));

      Dim d;
      std::getline(f, line);
      std::istringstream iss(line);
      iss >> d;
      Parameter param = model.add_parameters(d, name);

      std::vector<float> params_lst;
      auto deserialize_tensor_lambda = [&] () {
        for (int k1 = 0; k1 < d.d[0]; ++k1) {
          // CHECK DIMENSION
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
      };
      if (d.nd == 2) {
        transpose_lambda();
      }
      TensorTools::SetElements(param.get_storage().values, params_order_lst);

      params_lst.resize(0);
      deserialize_tensor_lambda();
      params_order_lst = params_lst;
      if (d.nd == 2) {
        transpose_lambda();
      }
      TensorTools::SetElements(param.get_storage().g, params_order_lst);
      std::getline(f, line);
    } // while Parameter
    
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
      LookupParameter lookup_param = model.add_lookup_parameters(N, d, name);

      std::vector<float> lookup_params_lst;
      auto deserialize_tensor_lambda = [&] () {
        for (int k1 = 0; k1 < all_dim.d[0]; ++k1) {
          // CHECK DIMENSION
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
      };
      if (all_dim.nd == 2) {
        transpose_lambda();
      }
      TensorTools::SetElements(lookup_param.get_storage().all_values,
                               lookup_params_order_lst);

      lookup_params_lst.resize(0);
      deserialize_tensor_lambda();
      lookup_params_order_lst = lookup_params_lst;
      if (all_dim.nd == 2) {
        transpose_lambda();
      }
      TensorTools::SetElements(lookup_param.get_storage().all_grads,
                               lookup_params_order_lst);
      std::getline(f, line);
    } // while LookupParameter

    if (line.size()) {
      if (line != "#") {
        throw std::runtime_error("Invalid model file format.");
      }
    }
    f.close();
    meta_f.close();
  }

 private:
  std::string fn, fn_meta;
  long long offset = 0;
}; // class Pack

} // namespace dynet

#endif
