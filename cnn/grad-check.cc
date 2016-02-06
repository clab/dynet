#include "cnn/grad-check.h"

#include <cassert>
#include <iostream>

#include "cnn/model.h"
#include "cnn/cnn.h"
#include "cnn/tensor.h"

using namespace std;

namespace cnn {

bool CheckGrad(Model& m, ComputationGraph& g, int verbosity) {
  // Clear the parameters first
  const vector<Parameters*>& params = m.parameters_list();
  const vector<LookupParameters*>& lookup_params = m.lookup_parameters_list();
  for (auto pp : params)
    pp->clear();
  for (auto pp : lookup_params)
    pp->clear();

  // Perform forward and backward steps
  float alpha = 5e-4;
  g.forward();
  g.backward();

  // Check
  bool flag = false, curr_flag = false;
  for (auto pp : params) {
    if(verbosity > 1)
      cerr << endl << "PARAMETERS " << pp << endl;
    Parameters& p = *pp;
    size_t ts = p.dim.size();
    for (size_t i = 0; i < ts; ++i) {
      float old = TensorTools::AccessElement(p.values, i);
      TensorTools::SetElement(p.values, i, old - alpha);
      float E_left = as_scalar(g.forward());
      TensorTools::SetElement(p.values, i, old + alpha);
      float E_right = as_scalar(g.forward());
      TensorTools::SetElement(p.values, i, old);
      float g = (E_right - E_left) / (2 * alpha);
      float g_act = TensorTools::AccessElement(p.g, i);
      float f = fabs(g - g_act);
      float m = max(fabs(g), fabs(g_act));
      if (f > 0.1 && m > 0.f) f /= m;
      if (f > 0.1 || std::isnan(f)) { flag = true; if(verbosity > 0) { curr_flag = true; cerr << "***[" << f << "] "; } }
      if(verbosity + (curr_flag ? 1 : 0) > 1) {
        cerr << g_act << ' ' << g << endl;
        curr_flag = false;
      }
    }
  }

  for (auto pp : lookup_params) {
    if(verbosity > 1)
      cerr << endl << "LOOKUP PARAMETERS " << pp << endl;
    LookupParameters& p = *pp;
    size_t ts = p.dim.size();
    for (unsigned j : p.non_zero_grads) {
      if(verbosity > 1)
        cerr << "OBJECT=" << j << endl;
      Tensor& v = p.values[j];
      Tensor& ag = p.grads[j];
      for (size_t i = 0; i < ts; ++i) {
        float old = TensorTools::AccessElement(v, i);
        TensorTools::SetElement(v, i, old - alpha);
        float E_left = as_scalar(g.forward());
        TensorTools::SetElement(v, i, old + alpha);
        float E_right = as_scalar(g.forward());
        TensorTools::SetElement(v, i, old);
        float g = (E_right - E_left) / (2 * alpha);
        float g_act = TensorTools::AccessElement(ag, i);
        float f = fabs(g - g_act);
        float m = max(fabs(g), fabs(g_act));
        if (f > 0.1 && m > 0.f) f /= m;
        if (f > 0.1 || std::isnan(f)) { flag = true; if(verbosity > 0) { curr_flag = true; cerr << "***[" << f << "] "; } }
        if(verbosity + (curr_flag ? 1 : 0) > 1) {
          cerr << g_act << ' ' << g << endl;
          curr_flag = false;
        }
      }
    }
  }

  if (flag) {
    if (verbosity > 1)
      cerr << endl << "*** GRADIENT CHECK FAILED ***" << endl;
  } else {
    if (verbosity > 0)
      cerr << endl << "GRADIENT CHECK PASSED" << endl;
  }
  return !flag;
}

}

