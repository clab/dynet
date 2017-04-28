#ifndef DYNET_STR_UTIL_H_
#define DYNET_STR_UTIL_H_

#include <vector>
#include <string>

namespace dynet {

inline bool startswith(const std::string & str, const std::string & key) {
  return str.find(key) == 0;
}

inline bool endswith(const std::string & str, const std::string & key) {
  if (str.size() < key.size()) return false;
  return str.rfind(key) == (str.size() - key.size());
}

inline std::vector<std::string> str_split(const std::string & str, const char sep) {
  std::vector<std::string> lst;
  size_t st = 0, en = 0;
  while (1) {
    en = str.find(sep, st);
    auto s = str.substr(st, en - st);
    if(s.size()) lst.push_back(s);
    if(en == std::string::npos) break;
    st = en + 1;
  }
  return lst;
}

} // namespace dynet

#endif
