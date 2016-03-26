#include "cnn/pretrain.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include "cnn/dict.h"
#include "cnn/model.h"

using namespace std;

namespace cnn {

void SavePretrainedEmbeddings(const std::string& fname,
    const Dict& d,
    const LookupParameter& lp) {
  cerr << "Writing word vectors to " << fname << " ...\n";
  ofstream out(fname);
  assert(out);
  auto& m = *lp.get();
  for (unsigned i = 0; i < d.size(); ++i) {
    out << d.Convert(i) << ' ' << (*m.values[i]).transpose() << endl;
  }
}

void ReadPretrainedEmbeddings(const std::string& fname,
    Dict* d,
    std::unordered_map<int, std::vector<float>>* vectors) {
  int unk = -1;
  if (d->is_frozen()) unk = d->GetUnkId();
  cerr << "Loading word vectors from " << fname << " ...\n";
  ifstream in(fname);
  assert(in);
  string line;
  string word;
  vector<float> v;
  getline(in, line);
  istringstream lin(line);
  lin >> word;
  while(lin) {
    float x;
    lin >> x;
    if (!lin) break;
    v.push_back(x);
  }
  unsigned vec_size = v.size();
  int wid = d->Convert(word);
  if (wid != unk) (*vectors)[wid] = v;
  while(getline(in, line)) {
    istringstream lin(line);
    lin >> word;
    int w = d->Convert(word);
    if (w != unk) {
      for (unsigned i = 0; i < vec_size; ++i) lin >> v[i];
      (*vectors)[w] = v;
    }
  }
}

} // cnn
