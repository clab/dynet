#include "dict.h"

#include <string>
#include <vector>
#include <sstream>

#include <boost/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#if BOOST_VERSION >= 105600
#include <boost/serialization/unordered_map.hpp>
#endif

#include "dynet/io-macros.h"


using namespace std;

namespace dynet {

std::vector<int> read_sentence(const std::string& line, Dict& sd) {
  std::istringstream in(line);
  std::string word;
  std::vector<int> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(sd.convert(word));
  }
  return res;
}

void read_sentence_pair(const std::string& line, std::vector<int>& s, Dict& sd, std::vector<int>& t, Dict& td) {
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  Dict* d = &sd;
  std::vector<int>* v = &s;
  while(in) {
    in >> word;
    if (!in) break;
    if (word == sep) { d = &td; v = &t; continue; }
    v->push_back(d->convert(word));
  }
}

template<class Archive> void Dict::serialize(Archive& ar, const unsigned int) {
#if BOOST_VERSION >= 105600
  ar & frozen;
  ar & map_unk;
  ar & unk_id;
  ar & words_;
  ar & d_;
#else
  throw std::invalid_argument("Serializing dictionaries is only supported on versions of boost 1.56 or higher");
#endif
}
DYNET_SERIALIZE_IMPL(Dict)

} // namespace dynet

