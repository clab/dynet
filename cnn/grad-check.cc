#include "cnn/grad-check.h"

#include <cassert>
#include <iostream>

#include "cnn/model.h"
#include "cnn/cnn.h"
#include "cnn/tensor.h"

using namespace std;

// 10e-5
#define GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL 5

namespace cnn {

void CheckGrad(Model& m, ComputationGraph& g) {
  float alpha = 5e-2; /// change to this alpha, which then shows that the difference between numeric and error propagation is around 10e-5.
  const vector<Parameters*>& ppparams = m.parameters_list();

  /// reset gradients to zero
  for (auto pp : ppparams)
  {
    Parameters& p = *pp;
    p.clear();
  }

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
      float threshold = (float)pow(10.0,
          max((float)0.0, ceil(log10(min(fabs(g), fabs(p.g.v[i]))))) - (int)GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL);
      float diff = fabs(g - p.g.v[i]);
      bool wrong = (std::isnan(diff) || diff > threshold);
        
      if (wrong)
      {
          flag = true; cerr << "***[" << diff << "] ";
      }
      cerr << p.g.v[i] << ' ' << g << endl;
    }
  }

  const vector<LookupParameters*>& lookup_params = m.lookup_parameters_list();
  for (auto pp : lookup_params) {
    cerr << "\nLOOKUP PARAMETERS " << pp << endl;
    LookupParameters& p = *pp;
    size_t ts = p.dim.size();
    for (unsigned j : p.non_zero_grads) {
      cerr << "OBJECT=" << j << endl;
      Tensor& v = p.values[j];
      Tensor& ag = p.grads[j];
      for (size_t i = 0; i < ts; ++i) {
        float old = v.v[i];
        v.v[i] = old - alpha;
        float E_left = as_scalar(g.forward());

        v.v[i] = old + alpha;
        float E_right = as_scalar(g.forward());
        float g = (E_right - E_left) / (2 * alpha);
        float f = fabs(g - ag.v[i]);
        float m = max(fabs(g), fabs(ag.v[i]));
        if (f > 0.1) {
          if (m > 0.f) f /= m;
          if (f > 0.1) { flag = true; cerr << "*** "; }
        }
        cerr << ag.v[i] << ' ' << g << endl;
      }
    }
  }

  if (flag) {
    cerr << "\n*** GRADIENT CHECK FAILED ***\n";
  } else {
    cerr << "\nGRADIENT CHECK PASSED\n";
  }
}

}

