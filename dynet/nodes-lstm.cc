#include "dynet/nodes-lstm.h"
#include "dynet/cpu-matrix-multiply.h"

#include "dynet/functors.h"
#include "dynet/simd-functors.h"
#include "dynet/nodes-macros.h"

using namespace std;

namespace dynet {


// ************* LSTM Gates *************

#ifndef __CUDACC__

  string VanillaLSTMGates::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "vanilla_lstm_gates(" << arg_names[0] << ", " << arg_names[1] << ", " << arg_names[2] << ", " << arg_names[3] << ", " << arg_names[4] << ')';
    return s.str();
  }

  Dim VanillaLSTMGates::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 5, "Failed input count check in VanillaLSTMGates");
    DYNET_ARG_CHECK(xs[0].ndims() == 1, "VanillaLSTMGates: x_t expected to be a vector");
    DYNET_ARG_CHECK(xs[1].ndims() == 1, "VanillaLSTMGates: h_tm1 expected to be a vector");
    DYNET_ARG_CHECK(xs[2].ndims() == 2, "VanillaLSTMGates: Wx expected to be a matrix");
    DYNET_ARG_CHECK(xs[3].ndims() == 2, "VanillaLSTMGates: Wh expected to be a matrix");
    DYNET_ARG_CHECK(xs[4].ndims() == 1, "VanillaLSTMGates: b expected to be a vector");
    unsigned hidden_dim=xs[1][0];
    unsigned input_dim=xs[0][0];
    unsigned batch_size=xs[0].bd;
    DYNET_ARG_CHECK(xs[2][0] == hidden_dim * 4, "VanillaLSTMGates: Wx dim 0 expected " << hidden_dim * 4 << ", was " << xs[2][0]);
    DYNET_ARG_CHECK(xs[2][1] == input_dim, "VanillaLSTMGates: Wx dim 1 expected " << input_dim << ", was " << xs[2][1]);
    DYNET_ARG_CHECK(xs[3][0] == hidden_dim * 4, "VanillaLSTMGates: Wh dim 0 expected " << hidden_dim * 4 << ", was " << xs[3][0]);
    DYNET_ARG_CHECK(xs[3][1] == hidden_dim, "VanillaLSTMGates: Wh dim 1 expected " << hidden_dim << ", was " << xs[3][1]);
    DYNET_ARG_CHECK(xs[4][0] == hidden_dim * 4, "VanillaLSTMGates: b dim expected " << hidden_dim * 4 << ", was " << xs[4][0]);
    return Dim({hidden_dim*4}, batch_size);
  }

#endif

  template<class MyDevice>
  void VanillaLSTMGates::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
    // computes affine transforms + nonlinearity
    // gates_i = sigmoid (Wx_i * x_t + Wh_i * h_tm1 + b_i)
    // gates_f = sigmoid (Wx_f * x_t + Wh_f * h_tm1 + b_f + 1)
    // gates_o = sigmoid (Wx_o * x_t + Wh_o * h_tm1 + b_o)
    // gates_g =   tanh  (Wx_g * x_t + Wh_g * h_tm1 + b_g)

    DYNET_ASSERT(xs.size() == 5, "Failed dimension check in VanillaLSTMGates::forward");

    const Tensor *x_t = xs[0];
    const Tensor *h_tm1 = xs[1];
    const Tensor *Wx = xs[2];
    const Tensor *Wh = xs[3];
    const Tensor *b  = xs[4];

    unsigned hidden_dim = b->d[0] / 4;
    unsigned batch_size = x_t->d.bd;

    Eigen::DSizes<ptrdiff_t, 2> indices_i(0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::DSizes<ptrdiff_t, 2> sizes_3(hidden_dim*3, static_cast<ptrdiff_t>(fx.d.bd));

    //bias
    Eigen::array<int, 3> bcast = {1, 1, (int)batch_size};
    fx.tbvec().device(*dev.edevice) = b->tbvec().broadcast(bcast);

    //matrix mult
    // TODO: this line will need special treatment on GPU
    // TODO: this seems to misbehave when using minibatches..
    CPUMatrixMultiply(dev, *Wx, *x_t, fx, kSCALAR_ONE);
    CPUMatrixMultiply(dev, *Wh, *h_tm1, fx, kSCALAR_ONE);

    cout << "fx:" << fx.tvec() << "\n";
    // non-linearities
    fx.tbvec().slice(indices_i, sizes_3).device(*dev.edevice) = fx.tbvec().slice(indices_i, sizes_3).unaryExpr(scalar_logistic_sigmoid_op<float>());
    fx.tbvec().slice(indices_g, sizes_1).device(*dev.edevice) = fx.tbvec().slice(indices_g, sizes_1).tanh();
  }

  template<class MyDevice>
  void VanillaLSTMGates::backward_dev_impl(const MyDevice & dev,
                               const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
    unsigned hidden_dim = fx.d[0] / 4;
    unsigned input_dim = xs[0]->d[0];
    unsigned batch_size = xs[0]->d.bd;
    Eigen::DSizes<ptrdiff_t, 3> indices_mat_i(0, 0, 0);
    Eigen::DSizes<ptrdiff_t, 3> indices_mat_g(hidden_dim*3, 0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_i(0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_o(hidden_dim*2,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 1> indices_i_nobatch(0);
    Eigen::DSizes<ptrdiff_t, 2> indices_mat_i_nobatch(0,0);
    Eigen::DSizes<ptrdiff_t, 1> indices_f_nobatch(hidden_dim);
    Eigen::DSizes<ptrdiff_t, 1> indices_o_nobatch(hidden_dim*2);
    Eigen::DSizes<ptrdiff_t, 1> indices_g_nobatch(hidden_dim*3);
    Eigen::DSizes<ptrdiff_t, 2> indices_mat_g_nobatch(hidden_dim*3, 0);
    Eigen::DSizes<ptrdiff_t, 3> sizes_mat_1(hidden_dim, 1, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::DSizes<ptrdiff_t, 3> sizes_mat_3(hidden_dim*3, 1, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::array<int, 1> vec_batch_axis; vec_batch_axis[0] = 1;
    Eigen::array<int, 2> mat_batch_axis; mat_batch_axis[0] = 1; mat_batch_axis[1] = 3;  // TODO: not sure why we have the extra dimension "1" after the outer product..

    Eigen::array<ptrdiff_t, 3> transp_order = {1,0,2};

    array<Eigen::IndexPair<int>, 1> product_mat = { Eigen::IndexPair<int>(1, 0) }; // following https://stackoverflow.com/questions/39815869/how-to-transpose-tensor-in-eigen
    cout << "doing anythin?\n";
    if(i==0){
      Eigen::DSizes<ptrdiff_t, 1> sizes_1_nobatch(input_dim);
      Eigen::DSizes<ptrdiff_t, 1> sizes_3_nobatch(input_dim*3);
      Eigen::DSizes<ptrdiff_t, 2> sizes_1(input_dim, static_cast<ptrdiff_t>(fx.d.bd));
      Eigen::DSizes<ptrdiff_t, 2> sizes_3(input_dim*3, static_cast<ptrdiff_t>(fx.d.bd));

      Eigen::array<int, 3> bcast; bcast[0] = 1; bcast[1] = 1; bcast[2] = batch_size;

      // dx_t = Wx_i^T * [di . i_t . (1-i_t)]
      //      + Wx_f^T * [df . f_t . (1-f_t)]
      //      + Wx_o^T * [do . o_t . (1-o_t)]
      //      + Wx_g^T * [dg . (1-tanh(g_t))]
      // note: here Wx is broadcasted over batches

      // TODO: fix/test
      // first handle the sigmoids
      dEdxi.tbvec().slice(indices_i, sizes_3).device(*dev.edevice) += (dEdf.tb<2>() * fx.tb<2>() * (fx.tb<2>().constant(1) - fx.tb<2>())).slice(indices_mat_i, sizes_mat_3).contract(xs[2]->tb<2>().broadcast(bcast).shuffle(transp_order), product_mat);
      cout << "worked!\n";

      // finally, the tanh
      // TODO
    } else if(i==1){ // dh_tm1
      // TODO: implement (math analogous to dx_t)
      Eigen::DSizes<ptrdiff_t, 1> sizes_1_nobatch(hidden_dim);
      Eigen::DSizes<ptrdiff_t, 1> sizes_3_nobatch(hidden_dim*3);

    } else if(i==2){
      // dWx_i = [di . i_t . (1-i_t)] * x_t (here * is outer product), then sum over batches
      // dWx_f = [di . f_t . (1-f_t)] * x_t (here * is outer product), then sum over batches
      // dWx_o = [di . o_t . (1-o_t)] * x_t (here * is outer product), then sum over batches
      Eigen::DSizes<ptrdiff_t, 2> sizes_mat_3_nobatch(hidden_dim*3, input_dim);
      dEdxi.t<2>().slice(indices_mat_i_nobatch, sizes_mat_3_nobatch).device(*dev.edevice) += (dEdf.tb<2>() * fx.tb<2>() * (fx.tb<2>().constant(1) - fx.tb<2>())).slice(indices_mat_i, sizes_mat_3).contract(xs[0]->tb<2>().shuffle(transp_order), product_mat).sum(mat_batch_axis);

      // dWx_g = [dg . (1-tanh(g_t))] * x_t (here * is outer product), then sum over batches
      Eigen::DSizes<ptrdiff_t, 2> sizes_mat_1_nobatch(hidden_dim, input_dim);
      dEdxi.t<2>().slice(indices_mat_g_nobatch, sizes_mat_1_nobatch).device(*dev.edevice) += (dEdf.tb<2>() * (fx.tb<2>().constant(1) - fx.tb<2>().tanh())).slice(indices_mat_g, sizes_mat_1).contract(xs[0]->tb<2>().shuffle(transp_order), product_mat).sum(mat_batch_axis);

    } else if(i==3){ // dWh
      // dWh_i = [di . i_t . (1-i_t)] * h_tm1 (here * is outer product), then sum over batches
      // dWh_f = [df . f_t . (1-f_t)] * h_tm1 (here * is outer product), then sum over batches
      // dWh_o = [do . o_t . (1-o_t)] * h_tm1 (here * is outer product), then sum over batches
      Eigen::DSizes<ptrdiff_t, 2> sizes_mat_3_nobatch(hidden_dim*3, hidden_dim);
      dEdxi.t<2>().slice(indices_mat_i_nobatch, sizes_mat_3_nobatch).device(*dev.edevice) += (dEdf.tb<2>() * fx.tb<2>() * (fx.tb<2>().constant(1) - fx.tb<2>())).slice(indices_mat_i, sizes_mat_3).contract(xs[1]->tb<2>().shuffle(transp_order), product_mat).sum(mat_batch_axis);

      // dWh_g = [dg . (1-tanh(g_t))] * h_tm1 (here * is outer product), then sum over batches
      Eigen::DSizes<ptrdiff_t, 2> sizes_mat_1_nobatch(hidden_dim, hidden_dim);
      dEdxi.t<2>().slice(indices_mat_g_nobatch, sizes_mat_1_nobatch).device(*dev.edevice) += (dEdf.tb<2>() * (fx.tb<2>().constant(1) - fx.tb<2>().tanh())).slice(indices_mat_g, sizes_mat_1).contract(xs[1]->tb<2>().shuffle(transp_order), product_mat).sum(mat_batch_axis);

    } else if(i==4){
      Eigen::DSizes<ptrdiff_t, 1> sizes_1_nobatch(hidden_dim);
      Eigen::DSizes<ptrdiff_t, 1> sizes_3_nobatch(hidden_dim*3);
      Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));
      Eigen::DSizes<ptrdiff_t, 2> sizes_3(hidden_dim*3, static_cast<ptrdiff_t>(fx.d.bd));

      // db_i = di . i_t . (1-i_t), then sum over batches
      // db_f = df . f_t . (1-f_t), then sum over batches
      // db_o = do . f_t . (1-o_t), then sum over batches
      dEdxi.tvec().slice(indices_i_nobatch, sizes_3_nobatch).device(*dev.edevice) += (dEdf.tbvec().slice(indices_i, sizes_3) * fx.tbvec().slice(indices_i, sizes_3) * (fx.tbvec().slice(indices_i, sizes_3).constant(1) - fx.tbvec().slice(indices_i, sizes_3))).sum(vec_batch_axis);

      // db_g = dg . (1 - tanh(g_t), then sum over batches
      dEdxi.tvec().slice(indices_g_nobatch, sizes_1_nobatch).device(*dev.edevice) += (dEdf.tbvec().slice(indices_g, sizes_1) * (fx.tbvec().slice(indices_i, sizes_1).constant(1) - fx.tbvec().slice(indices_g, sizes_1).tanh())).sum(vec_batch_axis);
    }
  }

  DYNET_NODE_INST_DEV_IMPL(VanillaLSTMGates)


  // ************* LSTM Cell *************

#ifndef __CUDACC__

  string VanillaLSTMC::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "vanilla_lstm_c(" << arg_names[0] << ", " << arg_names[1] << ')';
    return s.str();
  }

  Dim VanillaLSTMC::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in VanillaLSTMC");
    DYNET_ARG_CHECK(xs[0].ndims() == 1, "VanillaLSTMC: c_tm1 expected to be a vector");
    DYNET_ARG_CHECK(xs[1].ndims() == 1, "VanillaLSTMC: gates_t expected to be a vector");
    DYNET_ARG_CHECK(xs[0].size()*4 == xs[1].size(), "VanillaLSTMC: gates_t expected 4 times as big as c_t, but " << xs[0].size() << "*4 != " << xs[1].size());
    DYNET_ARG_CHECK(xs[0].bd == xs[1].bd, "VanillaLSTMC: gates_t and c_t expected to have equal batch size, but " << xs[0].bd << " != " << xs[1].bd);
    return xs[0];
  }

#endif

  template<class MyDevice>
  void VanillaLSTMC::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
    // computes cell state (elementwise multiplication)
    // c_t = gates_i . gates_g + gates_f . c_tm1

    DYNET_ASSERT(xs.size() == 2, "Failed dimension check in VanillaLSTMC::forward");

    const Tensor *c_tm1 = xs[0];
    const Tensor *gates_t = xs[1];

    unsigned hidden_dim = c_tm1->d[0];
    unsigned batch_size = c_tm1->d.bd;

    Eigen::DSizes<ptrdiff_t, 2> indices_i(0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));

    fx.tbvec().device(*dev.edevice) = gates_t->tbvec().slice(indices_i, sizes_1) * gates_t->tbvec().slice(indices_g, sizes_1)
  			          + gates_t->tbvec().slice(indices_f, sizes_1) * c_tm1->tbvec();

  }

  template<class MyDevice>
  void VanillaLSTMC::backward_dev_impl(const MyDevice & dev,
                               const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
    unsigned hidden_dim = fx.d[0];

    Eigen::DSizes<ptrdiff_t, 2> indices_i(0,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));

    if(i==0){ // dc_tm1 = dc_t . f_t
      dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec() * xs[1]->tbvec().slice(indices_f, sizes_1);
    } else if(i==1){
      // di_t = dc_t . g_t
      dEdxi.tbvec().slice(indices_i, sizes_1).device(*dev.edevice) += dEdf.tbvec() * xs[1]->tbvec().slice(indices_g, sizes_1);
      // df_t = dc_t . c_tm1
      dEdxi.tbvec().slice(indices_f, sizes_1).device(*dev.edevice) += dEdf.tbvec() * xs[0]->tbvec();
      // dg_t = dc_t . i_t
      dEdxi.tbvec().slice(indices_g, sizes_1).device(*dev.edevice) += dEdf.tbvec() * xs[1]->tbvec().slice(indices_i, sizes_1);
    }
  }

  DYNET_NODE_INST_DEV_IMPL(VanillaLSTMC)

  // ************* LSTM State *************

#ifndef __CUDACC__

  string VanillaLSTMH::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "vanilla_lstm_h(" << arg_names[0] << ", " << arg_names[1] << ')';
    return s.str();
  }

  Dim VanillaLSTMH::dim_forward(const vector<Dim>& xs) const {
    DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in VanillaLSTMH");
    DYNET_ARG_CHECK(xs[0].ndims() == 1, "VanillaLSTMH: c_t expected to be a vector");
    DYNET_ARG_CHECK(xs[1].ndims() == 1, "VanillaLSTMH: gates_t expected to be a vector");
    DYNET_ARG_CHECK(xs[0].size()*4 == xs[1].size(), "VanillaLSTMH: gates_t expected 4 times as big as c_t, but " << xs[0].size() << "*4 != " << xs[1].size());
    DYNET_ARG_CHECK(xs[0].bd == xs[1].bd, "VanillaLSTMH: gates_t and c_t expected to have equal batch size, but " << xs[0].bd << " != " << xs[1].bd);
    return xs[0];
  }

#endif

  template<class MyDevice>
  void VanillaLSTMH::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
    // computes output state (elementwise multiplication)
    // h_t = gates_o . tanh(c_t)

    DYNET_ASSERT(xs.size() == 2, "Failed dimension check in VanillaLSTMH::forward");

    const Tensor *c_t = xs[0];
    const Tensor *gates_t = xs[1];

    unsigned hidden_dim = c_t->d[0];
    unsigned batch_size = c_t->d.bd;

    Eigen::DSizes<ptrdiff_t, 2> indices_o(hidden_dim*2,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(batch_size));

    fx.tbvec().device(*dev.edevice) = gates_t->tbvec().slice(indices_o, sizes_1) * c_t->tbvec().tanh();
  }

  template<class MyDevice>
  void VanillaLSTMH::backward_dev_impl(const MyDevice & dev,
                               const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
    unsigned hidden_dim = fx.d[0];
    unsigned batch_size = fx.d.bd;
    Eigen::DSizes<ptrdiff_t, 2> indices_o(hidden_dim*2,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(batch_size));

    if(i==0){ // dc_t = dh_t . o_t . (1 - tanh(tanh(c_t)))
      // TODO: gradient checks not passing in case of multiple batches
      dEdxi.tbvec().device(*dev.edevice) += dEdf.tbvec() * xs[1]->tbvec().slice(indices_o, sizes_1) * (xs[0]->tbvec().constant(1) - xs[0]->tbvec().tanh().tanh());
    } else if(i==1){ // do_t = dh_t . tanh(c_t)
      dEdxi.tbvec().slice(indices_o, sizes_1).device(*dev.edevice) += dEdf.tbvec() * xs[0]->tbvec().tanh();
    }
  }

  DYNET_NODE_INST_DEV_IMPL(VanillaLSTMH)


}
