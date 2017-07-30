#include "dynet/nodes-lstm.h"
#include "dynet/matrix-multiply.h"

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
    DYNET_ARG_CHECK(xs.size() == 5 || xs.size() == 7, "Failed input count check in VanillaLSTMGates");
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
    if(xs.size() == 7){
      DYNET_ARG_CHECK(xs[5].ndims() == 1, "VanillaLSTMGates: dropout_mask_x expected to be a vector");
      DYNET_ARG_CHECK(xs[6].ndims() == 1, "VanillaLSTMGates: dropout_mask_h expected to be a vector");
      DYNET_ARG_CHECK(xs[5].bd == batch_size, "VanillaLSTMGates: dropout_mask_x expected to have batch size " << batch_size << ", was " << xs[5].bd);
      DYNET_ARG_CHECK(xs[6].bd == batch_size, "VanillaLSTMGates: dropout_mask_h expected to have batch size " << batch_size << ", was " << xs[6].bd);
      DYNET_ARG_CHECK(xs[5][0] == input_dim, "VanillaLSTMGates: dropout_mask_x dim 1 expected " << input_dim << ", was " << xs[5][0]);
      DYNET_ARG_CHECK(xs[6][0] == hidden_dim, "VanillaLSTMGates: dropout_mask_h dim 1 expected " << hidden_dim << ", was " << xs[6][0]);
    }
    return Dim({hidden_dim*4}, batch_size);
  }

  int VanillaLSTMGates::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
    Sig s(nt::vanilla_lstm_gates);
    // Assume parameter vectors must be same
    if(dim.bd == 1) {
      s.add_dim(cg.nodes[args[0]]->dim);
      // s.add_dim(cg.nodes[args[1]]->dim); // not necessary, as will be the same
      s.add_node(args[2]);
      s.add_node(args[3]);
      s.add_node(args[4]);
      if(args.size() == 7) {
        s.add_node(args[5]);
        s.add_node(args[6]);
      }
    } else {
      for(auto nid : args) {
        const Dim & d = cg.nodes[nid]->dim;
        if(d.bd == 1)
          s.add_node(nid);
        else
          s.add_dim(d);
      }
    }
    return sm.get_idx(s);
  }
  
  std::vector<int> VanillaLSTMGates::autobatch_concat(const ComputationGraph & cg) const {
    vector<int> ret(args.size(), 0);
    if(dim.bd == 1) {
      ret[0] = ret[1] = 1;
    } else {
      for(size_t i = 0; i < ret.size(); ++i)
        ret[i] = (cg.nodes[args[i]]->dim.bd > 1);
    }
    return ret;
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
    const Tensor *h_tm1 = (Tensor*)xs[1];
    const Tensor *Wx = xs[2];
    const Tensor *Wh = xs[3];
    const Tensor *b  = xs[4];

    unsigned hidden_dim = h_tm1->d[0];
    unsigned input_dim = x_t->d[0];
    unsigned batch_size = x_t->d.bd;

    Eigen::DSizes<ptrdiff_t, 2> indices_i(0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::DSizes<ptrdiff_t, 2> sizes_3(hidden_dim*3, static_cast<ptrdiff_t>(fx.d.bd));

    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];

    //bias
    Eigen::array<int, 3> bcast = {1, 1, (int)batch_size};
    fx.tb<2>().device(*dev.edevice) = b->tb<2>().broadcast(bcast);
    // forget gate: bias + 1
    fx.tbvec().slice(indices_f, sizes_1).device(*dev.edevice) += fx.tbvec().slice(indices_f, sizes_1).constant(1);

    if(xs.size()==7){
      Tensor x_t_dropped(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      x_t_dropped.v = static_cast<float*>(scratch_allocator->allocate(x_t_dropped.d.size() * sizeof(float)));
      x_t_dropped.tvec().device(*dev.edevice) = x_t->tvec() * xs[5]->tvec();
      x_t = &x_t_dropped;
      Tensor h_tm1_dropped(Dim({hidden_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      h_tm1_dropped.v = static_cast<float*>(scratch_allocator->allocate(h_tm1_dropped.d.size() * sizeof(float)));
      h_tm1_dropped.tvec().device(*dev.edevice) = h_tm1->tvec() * xs[6]->tvec();
      h_tm1 = &h_tm1_dropped;
    }
    //matrix mult
    if(weightnoise_std > 0.f){
      Tensor Wx_noisy(Dim({hidden_dim*4, input_dim},1), nullptr, fx.device, fx.mem_pool);
      Wx_noisy.v = static_cast<float*>(scratch_allocator->allocate(Wx_noisy.d.size() * sizeof(float)));
      TensorTools::randomize_normal(Wx_noisy, 0, weightnoise_std);
      Wx_noisy.tvec().device(*dev.edevice) += Wx->tvec();

      Tensor Wh_noisy(Dim({hidden_dim*4, hidden_dim},1), nullptr, fx.device, fx.mem_pool);
      Wh_noisy.v = static_cast<float*>(scratch_allocator->allocate(Wh_noisy.d.size() * sizeof(float)));
      TensorTools::randomize_normal(Wh_noisy, 0, weightnoise_std);
      Wh_noisy.tvec().device(*dev.edevice) += Wh->tvec();

      Tensor b_noisy(Dim({hidden_dim*4, 1},1), nullptr, fx.device, fx.mem_pool);
      b_noisy.v = static_cast<float*>(scratch_allocator->allocate(b_noisy.d.size() * sizeof(float)));
      TensorTools::randomize_normal(b_noisy, 0, weightnoise_std);
      b_noisy.tvec().device(*dev.edevice) += b->tvec();

    } else {
      MatrixMultiply(dev, *Wx, *x_t, fx, kSCALAR_ONE);
      MatrixMultiply(dev, *Wh, *h_tm1, fx, kSCALAR_ONE);
    }

    // non-linearities
    fx.tbvec().slice(indices_i, sizes_3).device(*dev.edevice) = fx.tbvec().slice(indices_i, sizes_3).unaryExpr(scalar_logistic_sigmoid_op<float>());
    fx.tbvec().slice(indices_g, sizes_1).device(*dev.edevice) = fx.tbvec().slice(indices_g, sizes_1).tanh();

    scratch_allocator->free();
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
    Eigen::DSizes<ptrdiff_t, 3> indices_mat_i_inp(0, 0, 0);
    Eigen::DSizes<ptrdiff_t, 3> indices_mat_f_inp(input_dim*1, 0, 0);
    Eigen::DSizes<ptrdiff_t, 3> indices_mat_o_inp(input_dim*2, 0, 0);
    Eigen::DSizes<ptrdiff_t, 3> indices_mat_g_inp(input_dim*3, 0, 0);
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
    Eigen::DSizes<ptrdiff_t, 3> sizes_mat_1_inp(input_dim, 1, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::DSizes<ptrdiff_t, 3> sizes_mat_3(hidden_dim*3, 1, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::array<int, 1> vec_batch_axis; vec_batch_axis[0] = 1;
    Eigen::array<int, 1> mat_batch_axis; mat_batch_axis[0] = 2;

    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];

    if(i==0){
        // goal: dx_t = [Wx_i]^T   [di . i_t . (1-i_t)]
        //              [Wx_f]   * [df . f_t . (1-f_t)]
        //              [Wx_o]     [do . o_t . (1-o_t)]
        //              [Wx_g]     [dg . (1 - g_t^2)]
        //       note: here Wx is broadcasted over batches
        // allocate scratch mem mult_l, mult_r
        Tensor mult_r(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
        mult_r.v = static_cast<float*>(scratch_allocator->allocate(mult_r.d.size() * sizeof(float)));
        Tensor mult_y(Dim({input_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
        mult_y.v = static_cast<float*>(scratch_allocator->allocate(mult_y.d.size() * sizeof(float)));

        // mult_r = [di . i_t . (1-i_t)]
        //          [df . f_t . (1-f_t)]
        //          [do . o_t . (1-o_t)]
        //          [dg . (1 - g_t^2)]
        mult_r.tb<2>().slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_i, sizes_mat_3) * fx.tb<2>().slice(indices_mat_i, sizes_mat_3) * (fx.tb<2>().slice(indices_mat_i, sizes_mat_3).constant(1) - fx.tb<2>().slice(indices_mat_i, sizes_mat_3));
        mult_r.tb<2>().slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_g, sizes_mat_1) * (fx.tb<2>().slice(indices_mat_g, sizes_mat_1).constant(1) - fx.tb<2>().slice(indices_mat_g, sizes_mat_1).square());

        // dx_t += mult_l^T * mult_r
        if(xs.size()==7){
          TensorTools::zero(mult_y);
          MatrixTranspMultiplyAcc(dev, *xs[2], mult_r, mult_y);
          dEdxi.tvec().device(*dev.edevice) += mult_y.tvec() * xs[5]->tvec();
        } else {
          MatrixTranspMultiplyAcc(dev, *xs[2], mult_r, dEdxi);
        }

    } else if(i==1){ // dh_tm1
        // goal: dh_tm1 = [Wh_i]^T   [di . i_t . (1-i_t)]
        //                [Wh_f]   * [df . f_t . (1-f_t)]
        //                [Wh_o]     [do . o_t . (1-o_t)]
        //                [Wh_g]     [dg . (1 - g_t^2)]
        //       note: here Wh is broadcasted over batches

        // allocate scratch mem mult_l, mult_r
        Tensor mult_r(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
        mult_r.v = static_cast<float*>(scratch_allocator->allocate(mult_r.d.size() * sizeof(float)));
        Tensor mult_y(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
        mult_y.v = static_cast<float*>(scratch_allocator->allocate(mult_y.d.size() * sizeof(float)));

        // mult_r = [di . i_t . (1-i_t)]
        //          [df . f_t . (1-f_t)]
        //          [do . o_t . (1-o_t)]
        //          [dg . (1 - g_t^2)]
        mult_r.tb<2>().slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_i, sizes_mat_3) * fx.tb<2>().slice(indices_mat_i, sizes_mat_3) * (fx.tb<2>().slice(indices_mat_i, sizes_mat_3).constant(1) - fx.tb<2>().slice(indices_mat_i, sizes_mat_3));
        mult_r.tb<2>().slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_g, sizes_mat_1) * (fx.tb<2>().slice(indices_mat_g, sizes_mat_1).constant(1) - fx.tb<2>().slice(indices_mat_g, sizes_mat_1).square());

        // dx_t += mult_l * mult_r
        if(xs.size()==7){
          TensorTools::zero(mult_y);
          MatrixTranspMultiplyAcc(dev, *xs[3], mult_r, mult_y);
          dEdxi.tvec().device(*dev.edevice) += mult_y.tvec() * xs[6]->tvec();
        } else {
          MatrixTranspMultiplyAcc(dev, *xs[3], mult_r, dEdxi);
        }

    } else if(i==2){ // dWx
      // goal: dWx_i = [di . i_t . (1-i_t)] * x_t (here * is outer product), then sum over batches
      //       dWx_f = [di . f_t . (1-f_t)] * x_t (here * is outer product), then sum over batches
      //       dWx_o = [di . o_t . (1-o_t)] * x_t (here * is outer product), then sum over batches
      //       dWx_g = [dg . (1 - g_t^2)] * x_t (here * is outer product), then sum over batches

      // allocate scratch mem mult_l, mult_r, mult_y
      Tensor mult_l(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l.v = static_cast<float*>(scratch_allocator->allocate(mult_l.d.size() * sizeof(float)));

      // mult_l = [di . i_t . (1-i_t)]
      //          [df . f_t . (1-f_t)]
      //          [do . o_t . (1-o_t)]
      //          [dg . (1 - g_t^2)]
      mult_l.tb<2>().slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_i, sizes_mat_3) * fx.tb<2>().slice(indices_mat_i, sizes_mat_3) * (fx.tb<2>().slice(indices_mat_i, sizes_mat_3).constant(1) - fx.tb<2>().slice(indices_mat_i, sizes_mat_3));
      mult_l.tb<2>().slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_g, sizes_mat_1) * (fx.tb<2>().slice(indices_mat_g, sizes_mat_1).constant(1) - fx.tb<2>().slice(indices_mat_g, sizes_mat_1).square());

      const Tensor *x_t = xs[0];
      if(xs.size()==7){
        Tensor x_t_dropped(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
        x_t_dropped.v = static_cast<float*>(scratch_allocator->allocate(x_t_dropped.d.size() * sizeof(float)));
        x_t_dropped.tvec().device(*dev.edevice) = x_t->tvec() * xs[5]->tvec();
        x_t = &x_t_dropped;
      }

      // dWh += (mult_l * mult_r).sum_batches()
      MatrixMultiplyTranspAcc(dev, mult_l, *x_t, dEdxi);

    } else if(i==3){ // dWh
      // goal: dWh_i = [di . i_t . (1-i_t)] * h_tm1 (here * is outer product), then sum over batches
      //       dWh_f = [df . f_t . (1-f_t)] * h_tm1 (here * is outer product), then sum over batches
      //       dWh_o = [do . o_t . (1-o_t)] * h_tm1 (here * is outer product), then sum over batches
      //       dWh_g = [dg . (1 - g_t^2)] * h_tm1 (here * is outer product), then sum over batches

      // allocate scratch mem mult_l, mult_r, mult_y
      Tensor mult_l(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l.v = static_cast<float*>(scratch_allocator->allocate(mult_l.d.size() * sizeof(float)));

      // mult_l = [di . i_t . (1-i_t)]
      //          [df . f_t . (1-f_t)]
      //          [do . o_t . (1-o_t)]
      //          [dg . (1 - g_t^2)]
      mult_l.tb<2>().slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_i, sizes_mat_3) * fx.tb<2>().slice(indices_mat_i, sizes_mat_3) * (fx.tb<2>().slice(indices_mat_i, sizes_mat_3).constant(1) - fx.tb<2>().slice(indices_mat_i, sizes_mat_3));
      mult_l.tb<2>().slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = dEdf.tb<2>().slice(indices_mat_g, sizes_mat_1) * (fx.tb<2>().slice(indices_mat_g, sizes_mat_1).constant(1) - fx.tb<2>().slice(indices_mat_g, sizes_mat_1).square());

      const Tensor *h_tm1 = (Tensor*)xs[1];
      if(xs.size()==7){
        Tensor h_tm1_dropped(Dim({hidden_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
        h_tm1_dropped.v = static_cast<float*>(scratch_allocator->allocate(h_tm1_dropped.d.size() * sizeof(float)));
        h_tm1_dropped.tvec().device(*dev.edevice) = h_tm1->tvec() * xs[6]->tvec();
        h_tm1 = &h_tm1_dropped;
      }

      // dWh += (mult_l * mult_r).sum(batches)
      MatrixMultiplyTranspAcc(dev, mult_l, *h_tm1, dEdxi);

    } else if(i==4){
      Eigen::DSizes<ptrdiff_t, 1> sizes_1_nobatch(hidden_dim);
      Eigen::DSizes<ptrdiff_t, 1> sizes_3_nobatch(hidden_dim*3);
      Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));
      Eigen::DSizes<ptrdiff_t, 2> sizes_3(hidden_dim*3, static_cast<ptrdiff_t>(fx.d.bd));

      // db_i = di . i_t . (1-i_t), then sum over batches
      // db_f = df . f_t . (1-f_t), then sum over batches
      // db_o = do . f_t . (1-o_t), then sum over batches
      dEdxi.tvec().slice(indices_i_nobatch, sizes_3_nobatch).device(*dev.edevice) += (dEdf.tbvec().slice(indices_i, sizes_3) * fx.tbvec().slice(indices_i, sizes_3) * (fx.tbvec().slice(indices_i, sizes_3).constant(1) - fx.tbvec().slice(indices_i, sizes_3))).sum(vec_batch_axis);

      // db_g = dg . (1 - g_t^2), then sum over batches
      dEdxi.tvec().slice(indices_g_nobatch, sizes_1_nobatch).device(*dev.edevice) += (dEdf.tbvec().slice(indices_g, sizes_1) * (fx.tbvec().slice(indices_i, sizes_1).constant(1) - fx.tbvec().slice(indices_g, sizes_1).square())).sum(vec_batch_axis);
    }
    scratch_allocator->free();

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

  int VanillaLSTMC::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
    Sig s(nt::vanilla_lstm_h);
    s.add_dim(cg.nodes[args[0]]->dim);
    return sm.get_idx(s);
  }
  
  std::vector<int> VanillaLSTMC::autobatch_concat(const ComputationGraph & cg) const {
    return vector<int>(2, 1);
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

  int VanillaLSTMH::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
    Sig s(nt::vanilla_lstm_h);
    s.add_dim(cg.nodes[args[0]]->dim);
    return sm.get_idx(s);
  }
  
  std::vector<int> VanillaLSTMH::autobatch_concat(const ComputationGraph & cg) const {
    return vector<int>(2, 1);
  }

#endif

  template<class MyDevice>
  void VanillaLSTMH::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
    // computes output state (elementwise multiplication)
    // h_t = o_t . tanh(c_t)

    DYNET_ASSERT(xs.size() == 2, "Failed dimension check in VanillaLSTMH::forward");

    const Tensor *c_t = xs[0];
    const Tensor *gates_t = xs[1];

    unsigned hidden_dim = c_t->d[0];
    unsigned batch_size = c_t->d.bd;

    Eigen::DSizes<ptrdiff_t, 3> indices_o(hidden_dim*2,0,0);
    Eigen::DSizes<ptrdiff_t, 3> sizes_1(hidden_dim, 1, static_cast<ptrdiff_t>(batch_size));

    fx.tb<2>().device(*dev.edevice) = gates_t->tb<2>().slice(indices_o, sizes_1) * c_t->tb<2>().tanh();
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
    Eigen::DSizes<ptrdiff_t, 3> indices_o(hidden_dim*2,0,0);
    Eigen::DSizes<ptrdiff_t, 3> sizes_1(hidden_dim, 1, static_cast<ptrdiff_t>(batch_size));

    if(i==0){
      // dc_t = dh_t . o_t . (1 - tanh^2(c_t)))
      //      = dh_t . o_t . (1 - (h_t cdiv o_t)^2)
      dEdxi.tb<2>().device(*dev.edevice) += dEdf.tb<2>()
                                            * xs[1]->tb<2>().slice(indices_o, sizes_1)
                                            * (xs[0]->tb<2>().constant(1) - xs[0]->tb<2>().tanh().square());
      // TODO: we could use the below..
      // - pro: potential speed up (replace tanh by cdiv)
      // - con: potential (though unlikely) division by 0
      //      dEdxi.tb<2>().device(*dev.edevice) += dEdf.tb<2>()
      //                                            * xs[1]->tb<2>().slice(indices_o, sizes_1)
      //                                            * (xs[0]->tb<2>().constant(1) - (fx.tb<2>() / xs[1]->tb<2>().slice(indices_o, sizes_1)).square());
    } else if(i==1){
      // do_t = dh_t . tanh(c_t)
      dEdxi.tb<2>().slice(indices_o, sizes_1).device(*dev.edevice) += dEdf.tb<2>() * xs[0]->tb<2>().tanh();
    }
  }

  DYNET_NODE_INST_DEV_IMPL(VanillaLSTMH)


}
