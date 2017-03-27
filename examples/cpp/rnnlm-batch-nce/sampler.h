#ifndef _SPARSE_SAMPLER_H_
#define _SPARSE_SAMPLER_H_

#include <random>
#include <string>
#include <map>

#include "dynet/dict.h"
#include "dynet/globals.h"

using namespace std;
using namespace dynet;

/**
 * Wrapper around Dict that also maintains unigram counts
 *
 * Stops counting when frozen
 */
struct Counter {

  vector<unsigned> counts;
  Dict dict;

  unsigned convertAndAdd(string word) {
    unsigned index = dict.convert(word);
    if(!dict.is_frozen()) {
      while(counts.size() <= index)
        counts.push_back(0);
      counts[index]++;
    }
    return index;
  }

  string convert(const int& id) {
    return dict.convert(id);
  }
  
  unsigned convert(string word) {
    return dict.convert(word);
  }

  unsigned size() {
    return dict.size();
  }

  unsigned freq(int index) {
    return counts.at(index);
  }
  
  const vector<unsigned>& getCounts() {
    return counts;
  }

  Dict& getDict() {
    return dict;
  }

  void freeze() {
    dict.freeze();
  }
  
};

/**
 * Wrapper around discrete_distribution that allows for sampling
 * bags of items from the distribution
 */
struct Sampler {

  discrete_distribution<> distn;
  unsigned distnSize;
  
  Sampler(const vector<unsigned>& counts)
    : distn(counts.begin(), counts.end())
  {
    distnSize = counts.size();
  }

  int sample() {
    return distn(*dynet::rndeng);
  }

  unsigned size() {
    return distnSize;
  }

  void sampleBag(unsigned tries, vector<unsigned>& indices, vector<float>& counts, vector<float>& probs) {
    map<unsigned,unsigned> sample_counts;
    for(unsigned i=0;i<tries;i++) {
      int samp = this->sample();
      sample_counts[samp]++;
    }
    for (auto& p : sample_counts) {
      indices.push_back(p.first);
      counts.push_back(p.second);
      probs.push_back(distn.probabilities()[p.first]);
    }
  }

  float prob(int index) {
    return distn.probabilities()[index];
  }

  vector<float> getProbs(const vector<unsigned>& indexes) {
    vector<float> probs;
    for(unsigned indx : indexes) {
      probs.push_back(distn.probabilities()[indx]);
    }
    return probs;
  }
};

#endif
