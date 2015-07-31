#include "cnn/grad-check.h"

#include <cassert>
#include <iostream>

#include "cnn/model.h"
#include "cnn/cnn.h"
#include "cnn/tensor.h"

using namespace std;

namespace cnn {

void CheckGrad(Model& m, ComputationGraph& g) {
  float alpha = 1e-4;
  float E = as_scalar(g.forward());
  g.backward();

  bool flag = false;
  const vector<Parameters*>& params = m.parameters_list();
  for (auto pp : params) {
    cerr << "\nPARAMETERS " << pp << endl;
    Parameters& p = *pp;
    size_t ts = p.dim.size();
    for (size_t i = 0; i < ts; ++i) {
      float old = p.values.v[i];
      p.values.v[i] = old - alpha;
      float E_left = as_scalar(g.forward());

      p.values.v[i] = old + alpha;
      float E_right = as_scalar(g.forward());
      float g = (E_right - E_left) / (2 * alpha);
      float f = fabs(g - p.g.v[i]);
      float m = max(fabs(g), fabs(p.g.v[i]));
      if (m > 0.f) f /= m;
      if (f > 0.1) { flag = true; cerr << "*** "; }
      cerr << p.g.v[i] << ' ' << g << endl;
    }
  }
  if (flag) {
    cerr << "\n*** GRADIENT CHECK FAILED ***\n";
  } else {
    cerr << "\nGRADIENT CHECK PASSED\n";
  }
//  const vector<LookupParameters*>& lookup_params = m.lookup_parameters_list();
}

}

