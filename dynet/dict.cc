#include "dict.h"

#include <string>
#include <vector>
#include <sstream>

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

} // namespace dynet

