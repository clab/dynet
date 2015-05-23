#include "cnn/graph.h"
#include "cnn/cnn.h"
#include <vector>

using namespace std;

namespace cnn {

void GraphOptimize(ComputationGraph* cg) {
  // topo sort
  vector<Node*>& nodes = cg->nodes;
  vector<int> longest_paths(nodes.size());
  for (unsigned i = 0; i < nodes.size(); ++i) {
    auto& v = *nodes[i];  // vertex v_i
    auto& lp = longest_paths[i]; // distance to v_i
    for (auto e : v.args) {
      int weight = 0;
      if (v.args.size() == 7) weight = 1;
      int pte = longest_paths[e] + weight;
      if (pte > lp) lp = pte;
    }
  }
  for (unsigned i = 0; i < nodes.size(); ++i) {
    vector<string> x;
    for (auto e : nodes[i]->args) {
      x.push_back(string("x") + to_string(e));
    }
    cerr << "LONGEST PATH: " << longest_paths[i] << "\tx" << i << " = " << nodes[i]->as_string(x) << endl;
  }
  abort();
}


} // namespace cnn
