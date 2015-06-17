#ifndef CNN_DICT_H_
#define CNN_DICT_H_

#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

#include <boost/version.hpp>
#if BOOST_VERSION >= 105600
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#endif

namespace cnn {

class Dict {
 typedef std::unordered_map<std::string, int> Map;
 public:
  Dict() : frozen(false) {
  }

  inline unsigned size() const { return words_.size(); }

  void Freeze() { frozen = true; }

  inline int Convert(const std::string& word) {
    auto i = d_.find(word);
    if (i == d_.end()) {
      if (frozen) {
        std::cerr << "Unknown word encountered: " << word << std::endl;
        abort();
      }
      words_.push_back(word);
      return d_[word] = words_.size() - 1;
    } else {
      return i->second;
    }
  }

  inline const std::string& Convert(const int& id) const {
    assert(id < (int)words_.size());
    return words_[id];
  }

  void clear() { words_.clear(); d_.clear(); }

 private:
  bool frozen;
  std::vector<std::string> words_;
  Map d_;

#if BOOST_VERSION >= 105600
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & frozen;
    ar & words_;
    ar & d_;
  }
#endif
};

std::vector<int> ReadSentence(const std::string& line, Dict* sd);
void ReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td);

} // namespace cnn

#endif
