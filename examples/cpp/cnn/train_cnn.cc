#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/grad-check.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;

void check(Model& m, expr::Expression& expr, int verbosity) {
  ComputationGraph& g = *expr.pg;
  // Clear the parameters first
  const vector<ParameterStorage*>& params = m.parameters_list();
  const vector<LookupParameterStorage*>& lookup_params = m.lookup_parameters_list();
  for (auto pp : params)
    pp->clear();
  for (auto pp : lookup_params)
    pp->clear();

  // Perform forward and backward steps
  g.forward(expr);
  g.backward(expr);
  //float alpha = 5e-6;
  // cout.precision(20);
  //for (int i = 0; i < 1; ++i) {
  //  auto act = as_vector(g.forward(expr));
  //  for (int j = 0; j < act.size(); ++j) {
  //    cout << act[j] << "\t";
  //  }
  //  //float act = as_scalar(g.forward(expr));
  ////g.backward(expr);
  //  cout << endl; 
  //}
  // Check
  //bool flag = false, curr_flag = false;
  /*for (auto pp : params) {
    ParameterStorage& p = *pp;
    if (true) {
    unsigned N = p.g.d[3];
    unsigned C = p.g.d[2];
    unsigned H = p.g.d[0];
    unsigned W = p.g.d[1];
    for (size_t n = 0; n < N; ++n)
      for (size_t c = 0; c < C; ++c)
        for (size_t h = 0; h < H; ++h)
          for (size_t w = 0; w < W; ++w) {
            float g_gt = TensorTools::access_element(p.g, n*C*H*W + c*H*W + w * H + h);
            //float g_gt = p.g.v[n*C*H*W + c*H*W + h * W + w];
            cout << "(" << n << "," << c << "," << h << "," << w <<"): " << g_gt << endl;
          }
          }
  }*/
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  // parameters
  Model m;
  Parameter param_kernel, param_bias, param_kernel2;
  unsigned fH = 2;
  unsigned fW = 2;
  unsigned H = 49;
  unsigned W = 49;
  param_kernel = m.add_parameters({fH, fW, 2, 2});
  param_kernel2 = m.add_parameters({fH, fW, 2, 2});
  std::vector<float> param_kernel_vals(fH * fW * 2 * 3);
  param_kernel_vals = {.011f, .022f, .033f, .012f, .022f, .032f, .013f, .023f, .033f,
                                         .111f, -.122f, -.033f, -.112f, -.022f, -.132f, -.113f, -.123f, -.133f,
                                         .211f, .222f, .233f, .212f, .222f, .232f};
  TensorTools::set_elements(param_kernel.get()->values, param_kernel_vals);
  TensorTools::set_elements(param_kernel2.get()->values, param_kernel_vals);
  std::vector<float> conv2d_batch_vals(2 * H * W * 2);
  for (unsigned i = 0; i < conv2d_batch_vals.size(); ++i) {
    conv2d_batch_vals[i] = i * 0.011f + (i+1) * 0.001f * 1000;
  }

  ComputationGraph cg;
  Expression x = input(cg, Dim({H, W, 2}, 2), conv2d_batch_vals); 
  Expression kernel = parameter(cg, param_kernel);
  //Expression bias = parameter(cg, param_bias);
  vector<unsigned> stride = {4, 4}; bool is_valid = false;
  Expression y1 = conv2d(x, kernel, stride, is_valid);

  Expression kernel2 = parameter(cg, param_kernel2);
  Expression y2 = conv2d(y1, kernel2, stride, is_valid);


  Expression y3 = sum_elems(y2);
  Expression y4 = sum_batches(sum_elems(y3));
  //check_grad(m, z1, 5);
  //check(m, y1, 5);
  check(m, y4, 5);
  //check(m, y3, 5);

  //Expression y2 = conv2d(x, kernel, bias, stride, is_valid);
  //Expression z2 = sum_batches(sum_elems(y2));
  //check_grad(m, z2, 5);

  //Expression y3 = conv2d(x, kernel, stride);
  //Expression z3 = sum_batches(sum_elems(y3));
  //check_grad(m, z3, 5);

  //Expression y4 = conv2d(x, kernel, bias, stride);
  //Expression z4 = sum_batches(sum_elems(y4));
  //check_grad(m, z4, 5);
  return 0;
}

