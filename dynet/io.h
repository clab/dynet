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
    /*
    auto lookup_params = model.get_lookup_parameter_storages();
    for (auto & lookup_param: lookup_params) {
    }
    */
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
      name = dynet::str_split(name, '_').front();

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
          params_lst.insert(tmp.end(), params_lst.begin(), params_lst.end());
        }
      };
      deserialize_tensor_lambda();
      TensorTools::SetElements(param.get_storage().values, params_lst);

      params_lst.resize(0);
      deserialize_tensor_lambda();
      TensorTools::SetElements(param.get_storage().g, params_lst);
      std::getline(f, line);
    } // while Parameter
    
    while (line == "#LookupParameter#") {
      // TODO
    }
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
