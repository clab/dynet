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
std::istream& operator>>(std::istream& is, std::vector<T> & v) {
  std::copy(std::istream_iterator<T>(is), std::istream_iterator<T>(), v.begin());
  return is;
}

class Pack {
 public:
  Pack(std::string filename) : fn(filename), fn_meta(filename + ".meta") {}
  void save(ParameterCollection & model, std::string key = "", bool is_append = false);
  void load(ParameterCollection & model, std::string key = "");
 
 private:
  bool duplicate_key_check(const std::string & key); 
  void serialize(ParameterCollection & model, std::string key, bool is_append);
  void deserialize(ParameterCollection & model, std::string key);

 private:
  std::string fn, fn_meta;
  long long offset = 0;
}; // class Pack

} // namespace dynet

#endif
