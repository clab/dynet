#include "dynet/tensor-eigen.h"
#include "dynet/nodes-lstm.h"
#include "dynet/matrix-multiply.h"

#include "dynet/functors.h"
#include "dynet/simd-functors.h"
#include "dynet/nodes-impl-macros.h"

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
    if(dropout){
      DYNET_ARG_CHECK(xs.size() >= 7, "Failed input count check in VanillaLSTMGates");
    }else{
      DYNET_ARG_CHECK(xs.size() >= 5, "Failed input count check in VanillaLSTMGates");
    }
    unsigned num_inputs = dropout?xs.size()-6:xs.size()-4;
    unsigned hidden_dim=xs[num_inputs][0];
    unsigned input_dim=xs[num_inputs+1][1];
    unsigned batch_size=xs[0].bd;
    unsigned inputs_dim_sum=0;
    for(unsigned i=0; i<num_inputs; i++){
        DYNET_ARG_CHECK(xs[i].ndims() == 1, "VanillaLSTMGates: x_t[" << i << "] expected to be a vector");
        DYNET_ARG_CHECK(xs[i].bd == batch_size, "VanillaLSTMGates: x_t has inconsistent batch size");
        inputs_dim_sum += xs[i][0];
    }
    DYNET_ARG_CHECK(inputs_dim_sum == input_dim, "VanillaLSTMGates: x_t has inconsistent dimension");
    DYNET_ARG_CHECK(xs[num_inputs].ndims() == 1, "VanillaLSTMGates: h_tm1 expected to be a vector");
    DYNET_ARG_CHECK(xs[num_inputs+1].ndims() == 2, "VanillaLSTMGates: Wx expected to be a matrix");
    DYNET_ARG_CHECK(xs[num_inputs+2].ndims() == 2, "VanillaLSTMGates: Wh expected to be a matrix");
    DYNET_ARG_CHECK(xs[num_inputs+3].ndims() == 1, "VanillaLSTMGates: b expected to be a vector");
    DYNET_ARG_CHECK(xs[num_inputs+1][0] == hidden_dim * 4, "VanillaLSTMGates: Wx dim 0 expected " << hidden_dim * 4 << ", was " << xs[2][0]);
    DYNET_ARG_CHECK(xs[num_inputs+1][1] == input_dim, "VanillaLSTMGates: Wx dim 1 expected " << input_dim << ", was " << xs[2][1]);
    DYNET_ARG_CHECK(xs[num_inputs+2][0] == hidden_dim * 4, "VanillaLSTMGates: Wh dim 0 expected " << hidden_dim * 4 << ", was " << xs[3][0]);
    DYNET_ARG_CHECK(xs[num_inputs+2][1] == hidden_dim, "VanillaLSTMGates: Wh dim 1 expected " << hidden_dim << ", was " << xs[3][1]);
    DYNET_ARG_CHECK(xs[num_inputs+3][0] == hidden_dim * 4, "VanillaLSTMGates: b dim expected " << hidden_dim * 4 << ", was " << xs[4][0]);
    if(dropout){
      DYNET_ARG_CHECK(xs[num_inputs+4].ndims() == 1, "VanillaLSTMGates: dropout_mask_x expected to be a vector");
      DYNET_ARG_CHECK(xs[num_inputs+5].ndims() == 1, "VanillaLSTMGates: dropout_mask_h expected to be a vector");
      DYNET_ARG_CHECK(xs[num_inputs+4].bd == batch_size || xs[num_inputs+4].bd == 1, "VanillaLSTMGates: dropout_mask_x expected to have batch size 1 or " << batch_size << ", was " << xs[5].bd);
      DYNET_ARG_CHECK(xs[num_inputs+5].bd == batch_size || xs[num_inputs+5].bd == 1, "VanillaLSTMGates: dropout_mask_h expected to have batch size 1 or " << batch_size << ", was " << xs[6].bd);
      DYNET_ARG_CHECK(xs[num_inputs+4][0] == input_dim, "VanillaLSTMGates: dropout_mask_x dim 1 expected " << input_dim << ", was " << xs[5][0]);
      DYNET_ARG_CHECK(xs[num_inputs+5][0] == hidden_dim, "VanillaLSTMGates: dropout_mask_h dim 1 expected " << hidden_dim << ", was " << xs[6][0]);
    }
    return Dim({hidden_dim*4}, batch_size);
  }

  int VanillaLSTMGates::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
    Sig s(nt::vanilla_lstm_gates);
    unsigned num_inputs = dropout?args.size()-6:args.size()-4;
    // Assume parameter vectors must be same
    if(dim.bd == 1) {
      for(unsigned i=0; i<num_inputs; i++)
        s.add_dim(cg.nodes[args[i]]->dim);
      // TODO: correct? parameter vectors would be args[num_inputs+1] .. args[num_inputs+3]
      // s.add_dim(cg.nodes[args[num_inputs]]->dim); // not necessary, as will be the same
      s.add_node(args[num_inputs+1]);
      s.add_node(args[num_inputs+2]);
      s.add_node(args[num_inputs+3]);
      if(dropout) {
        s.add_node(args[num_inputs+4]);
        s.add_node(args[num_inputs+5]);
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

    unsigned num_inputs = dropout?xs.size()-6:xs.size()-4;
    Tensor h_tm1(xs[num_inputs]->d, xs[num_inputs]->v, xs[num_inputs]->device, xs[num_inputs]->mem_pool);
    const Tensor *Wx = xs[num_inputs+1];
    const Tensor *Wh = xs[num_inputs+2];
    const Tensor *b  = xs[num_inputs+3];

    unsigned hidden_dim = h_tm1.d[0];
    unsigned input_dim = Wx->d[1];
    unsigned batch_size = xs[0]->d.bd;

    Eigen::DSizes<ptrdiff_t, 2> indices_i(0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));
    Eigen::DSizes<ptrdiff_t, 2> sizes_3(hidden_dim*3, static_cast<ptrdiff_t>(fx.d.bd));

    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];

    Tensor x_t(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
    if(num_inputs==1){
      x_t.v = xs[0]->v;
    } else {
      Tensor tmp(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      tmp.v = static_cast<float*>(scratch_allocator->allocate(tmp.d.size() * sizeof(float)));
      Eigen::DSizes<ptrdiff_t, 2> indices_tmp(0, 0);
      Eigen::DSizes<ptrdiff_t, 2> sizes_tmp(0, static_cast<ptrdiff_t>(fx.d.bd));
      for(unsigned i=0; i<num_inputs; i++){
        sizes_tmp[0] = xs[i]->d[0];
        tbvec(tmp).slice(indices_tmp, sizes_tmp).device(*dev.edevice) = tbvec(*xs[i]);
        indices_tmp[0] += xs[i]->d[0];
      }
      x_t.v = tmp.v;
    }

    //bias
#ifdef __CUDACC__
    Eigen::array<int, 3> bcast = {1, 1, (int)batch_size};
    tb<2>(fx).device(*dev.edevice) = tb<2>(*b).broadcast(bcast);
#else
    float *curr_ptr = fx.v, *end_ptr = curr_ptr + fx.d.size(), *in_ptr = b->v;
    do {
      memcpy(curr_ptr, in_ptr, sizeof(float)*b->d[0]);
      curr_ptr += b->d[0];
    } while(curr_ptr != end_ptr);
#endif
    tbvec(fx).slice(indices_f, sizes_1).device(*dev.edevice) += tbvec(fx).slice(indices_f, sizes_1).constant(forget_gate_bias);

    if(dropout){
      Tensor mask_x(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      if(xs[num_inputs+4]->d.bd == batch_size){
        mask_x.v = xs[num_inputs+4]->v;
      } else {
        mask_x.v = static_cast<float*>(scratch_allocator->allocate(mask_x.d.size() * sizeof(float)));
#ifdef __CUDACC__
        Eigen::array<int, 2> bcast = {1, (int)batch_size};
        tbvec(mask_x).device(*dev.edevice) = tbvec(*xs[num_inputs+4]).broadcast(bcast);
#else
        float *curr_ptr = mask_x.v, *end_ptr = curr_ptr + mask_x.d.size(), *in_ptr = xs[num_inputs+4]->v;
        do {
          memcpy(curr_ptr, in_ptr, sizeof(float)*xs[num_inputs+4]->d[0]);
          curr_ptr += xs[num_inputs+4]->d[0];
        } while(curr_ptr != end_ptr);
#endif
      }
      Tensor x_t_dropped(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      x_t_dropped.v = static_cast<float*>(scratch_allocator->allocate(x_t_dropped.d.size() * sizeof(float)));
      tvec(x_t_dropped).device(*dev.edevice) = tvec(x_t) * tvec(mask_x);
      x_t.v = x_t_dropped.v;

      Tensor mask_h(Dim({hidden_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      if(xs[num_inputs+5]->d.bd == batch_size){
        mask_h.v = xs[num_inputs+5]->v;
      } else {
        mask_h.v = static_cast<float*>(scratch_allocator->allocate(mask_h.d.size() * sizeof(float)));
#ifdef __CUDACC__
        Eigen::array<int, 2> bcast = {1, (int)batch_size};
        tbvec(mask_h).device(*dev.edevice) = tbvec(*xs[num_inputs+5]).broadcast(bcast);
#else
        float *curr_ptr = mask_h.v, *end_ptr = curr_ptr + mask_h.d.size(), *in_ptr = xs[num_inputs+5]->v;
        do {
          memcpy(curr_ptr, in_ptr, sizeof(float)*xs[num_inputs+5]->d[0]);
          curr_ptr += xs[num_inputs+5]->d[0];
        } while(curr_ptr != end_ptr);
#endif
      }
      Tensor h_tm1_dropped(Dim({hidden_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      h_tm1_dropped.v = static_cast<float*>(scratch_allocator->allocate(h_tm1_dropped.d.size() * sizeof(float)));
      tvec(h_tm1_dropped).device(*dev.edevice) = tvec(h_tm1) * tvec(mask_h);
      h_tm1.v = h_tm1_dropped.v;
    }
    //matrix mult
    if(weightnoise_std > 0.f){
      Tensor Wx_noisy(Dim({hidden_dim*4, input_dim},1), nullptr, fx.device, fx.mem_pool);
      Wx_noisy.v = static_cast<float*>(scratch_allocator->allocate(Wx_noisy.d.size() * sizeof(float)));
      TensorTools::randomize_normal(Wx_noisy, 0, weightnoise_std);
      tvec(Wx_noisy).device(*dev.edevice) += tvec(*Wx);

      Tensor Wh_noisy(Dim({hidden_dim*4, hidden_dim},1), nullptr, fx.device, fx.mem_pool);
      Wh_noisy.v = static_cast<float*>(scratch_allocator->allocate(Wh_noisy.d.size() * sizeof(float)));
      TensorTools::randomize_normal(Wh_noisy, 0, weightnoise_std);
      tvec(Wh_noisy).device(*dev.edevice) += tvec(*Wh);

      Tensor b_noisy(Dim({hidden_dim*4, 1},1), nullptr, fx.device, fx.mem_pool);
      b_noisy.v = static_cast<float*>(scratch_allocator->allocate(b_noisy.d.size() * sizeof(float)));
      TensorTools::randomize_normal(b_noisy, 0, weightnoise_std);
      tvec(b_noisy).device(*dev.edevice) += tvec(*b);

    } else {
      MatrixMultiply(dev, *Wx, x_t, fx, dev.kSCALAR_ONE);
      MatrixMultiply(dev, *Wh, h_tm1, fx, dev.kSCALAR_ONE);
    }

    // non-linearities
    Tensor fx_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
    fx_ifo.v = static_cast<float*>(scratch_allocator->allocate(fx_ifo.d.size() * sizeof(float)));
    tbvec(fx_ifo).device(*dev.edevice) = tbvec(fx).slice(indices_i, sizes_3);
    tbvec(fx_ifo).device(*dev.edevice) = tbvec(fx_ifo).unaryExpr(scalar_logistic_sigmoid_op<float>());
    tbvec(fx).slice(indices_i, sizes_3).device(*dev.edevice) = tbvec(fx_ifo);

    Tensor fx_g(Dim({hidden_dim*1, 1},batch_size), nullptr, fx.device, fx.mem_pool);
    fx_g.v = static_cast<float*>(scratch_allocator->allocate(fx_g.d.size() * sizeof(float)));
    tbvec(fx_g).device(*dev.edevice) = tbvec(fx).slice(indices_g, sizes_1);
    tbvec(fx_g).device(*dev.edevice) = tbvec(fx_g).tanh();
    tbvec(fx).slice(indices_g, sizes_1).device(*dev.edevice) = tbvec(fx_g);

    scratch_allocator->free();
  }

  template<class MyDevice>
  void VanillaLSTMGates::backward_dev_impl(const MyDevice & dev,
                               const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
    unsigned num_inputs = dropout?xs.size()-6:xs.size()-4;
    Tensor h_tm1(xs[num_inputs]->d, xs[num_inputs]->v, xs[num_inputs]->device, xs[num_inputs]->mem_pool);
    const Tensor *Wx = xs[num_inputs+1];
    const Tensor *Wh = xs[num_inputs+2];
//    const Tensor *b  = xs[num_inputs+3];

    unsigned hidden_dim = fx.d[0] / 4;
    unsigned input_dim = Wx->d[1];
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

    // scratch memory to avoid striding for element-wise operations
    Tensor dEdf_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
    dEdf_ifo.v = static_cast<float*>(scratch_allocator->allocate(dEdf_ifo.d.size() * sizeof(float)));
    Tensor dEdf_g(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
    dEdf_g.v = static_cast<float*>(scratch_allocator->allocate(dEdf_g.d.size() * sizeof(float)));
    Tensor fx_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
    fx_ifo.v = static_cast<float*>(scratch_allocator->allocate(fx_ifo.d.size() * sizeof(float)));
    Tensor fx_g(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
    fx_g.v = static_cast<float*>(scratch_allocator->allocate(fx_g.d.size() * sizeof(float)));
    tb<2>(dEdf_ifo).device(*dev.edevice) = tb<2>(dEdf).slice(indices_mat_i, sizes_mat_3);
    tb<2>(dEdf_g).device(*dev.edevice)   = tb<2>(dEdf).slice(indices_mat_g, sizes_mat_1);
    tb<2>(fx_ifo).device(*dev.edevice) = tb<2>(fx).slice(indices_mat_i, sizes_mat_3);
    tb<2>(fx_g).device(*dev.edevice)   = tb<2>(fx).slice(indices_mat_g, sizes_mat_1);

    Tensor mask_x(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
    Tensor mask_h(Dim({hidden_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
    if(i < num_inputs+3 && dropout){
      if(xs[num_inputs+4]->d.bd == batch_size){
        mask_x.v = xs[num_inputs+4]->v;
      } else {
        mask_x.v = static_cast<float*>(scratch_allocator->allocate(mask_x.d.size() * sizeof(float)));
#ifdef __CUDACC__
        Eigen::array<int, 2> bcast = {1, (int)batch_size};
        tbvec(mask_x).device(*dev.edevice) = tbvec(*xs[num_inputs+4]).broadcast(bcast);
#else
        float *curr_ptr = mask_x.v, *end_ptr = curr_ptr + mask_x.d.size(), *in_ptr = xs[num_inputs+4]->v;
        do {
          memcpy(curr_ptr, in_ptr, sizeof(float)*xs[num_inputs+4]->d[0]);
          curr_ptr += xs[num_inputs+4]->d[0];
        } while(curr_ptr != end_ptr);
#endif
      }
      if(xs[num_inputs+5]->d.bd == batch_size){
        mask_h.v = xs[num_inputs+5]->v;
      } else {
        mask_h.v = static_cast<float*>(scratch_allocator->allocate(mask_h.d.size() * sizeof(float)));
#ifdef __CUDACC__
        Eigen::array<int, 2> bcast = {1, (int)batch_size};
        tbvec(mask_h).device(*dev.edevice) = tbvec(*xs[num_inputs+5]).broadcast(bcast);
#else
        float *curr_ptr = mask_h.v, *end_ptr = curr_ptr + mask_h.d.size(), *in_ptr = xs[num_inputs+5]->v;
        do {
          memcpy(curr_ptr, in_ptr, sizeof(float)*xs[num_inputs+5]->d[0]);
          curr_ptr += xs[num_inputs+5]->d[0];
        } while(curr_ptr != end_ptr);
#endif
      }
    }


    if(i < num_inputs){
      // dx_t = [Wx_i]^T   [di . i_t . (1-i_t)]
      //        [Wx_f]   * [df . f_t . (1-f_t)]
      //        [Wx_o]     [do . o_t . (1-o_t)]
      //        [Wx_g]     [dg . (1 - g_t^2)]
      //    where Wx is broadcasted over batches

      // when using multiple inputs: slice out the one we're backpropagating to
      Tensor Wx_slice(Dim({hidden_dim*4, xs[i]->d[0]},1), nullptr, fx.device, fx.mem_pool);
      if(num_inputs==1){
        Wx_slice.v = Wx->v;
      } else {
        unsigned offset=0;
        for(unsigned j=0; j<i; j++) offset += xs[j]->d[0];
        Eigen::DSizes<ptrdiff_t, 3> indices_Wx(0, offset, 0);
        Eigen::DSizes<ptrdiff_t, 3> sizes_Wx(hidden_dim*4, xs[i]->d[0], 1);
        Wx_slice.v = static_cast<float*>(scratch_allocator->allocate(Wx_slice.d.size() * sizeof(float)));
        tb<2>(Wx_slice).device(*dev.edevice) = tb<2>(*Wx).slice(indices_Wx, sizes_Wx);
      }

      // scratch memory for the matrix multiplication
      Tensor mult_r_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_r_ifo.v = static_cast<float*>(scratch_allocator->allocate(mult_r_ifo.d.size() * sizeof(float)));
      Tensor mult_r_g(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_r_g.v = static_cast<float*>(scratch_allocator->allocate(mult_r_g.d.size() * sizeof(float)));
      Tensor mult_r(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_r.v = static_cast<float*>(scratch_allocator->allocate(mult_r.d.size() * sizeof(float)));

      // mult_r = [di . i_t . (1-i_t)]
      //          [df . f_t . (1-f_t)]
      //          [do . o_t . (1-o_t)]
      //          [dg . (1 - g_t^2)]
      tb<2>(mult_r_ifo).device(*dev.edevice) = tb<2>(dEdf_ifo) * tb<2>(fx_ifo) * (tb<2>(fx_ifo).constant(1) - tb<2>(fx_ifo));
      tb<2>(mult_r_g).device(*dev.edevice) = tb<2>(dEdf_g) * (tb<2>(fx_g).constant(1) - tb<2>(fx_g).square());
      tb<2>(mult_r).slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = tb<2>(mult_r_ifo);
      tb<2>(mult_r).slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = tb<2>(mult_r_g);
      // dx_t += mult_l^T * mult_r
      if(dropout){
        Tensor mult_y(Dim({xs[i]->d[0], 1},batch_size), nullptr, fx.device, fx.mem_pool);
        mult_y.v = static_cast<float*>(scratch_allocator->allocate(mult_y.d.size() * sizeof(float)));
        TensorTools::zero(mult_y);
        MatrixTranspMultiplyAcc(dev, Wx_slice, mult_r, mult_y);
        // when using multiple inputs: slice out the appropriate dropout mask
        Tensor dropout_mask(Dim({xs[i]->d[0]}, mask_x.d.bd), nullptr, fx.device, fx.mem_pool);
        if(num_inputs==1){
          dropout_mask.v = mask_x.v;
        } else {
          unsigned offset=0;
          for(unsigned j=0; j<i; j++) offset += xs[j]->d[0];
          Eigen::DSizes<ptrdiff_t, 2> indices_dropout(offset, 0);
          Eigen::DSizes<ptrdiff_t, 2> sizes_dropout(xs[i]->d[0], mask_x.d.bd);
          dropout_mask.v = static_cast<float*>(scratch_allocator->allocate(dropout_mask.d.size() * sizeof(float)));
          tbvec(dropout_mask).device(*dev.edevice) = tbvec(mask_x).slice(indices_dropout, sizes_dropout);
        }
        tvec(dEdxi).device(*dev.edevice) += tvec(mult_y) * tvec(dropout_mask);
      } else {
		    MatrixTranspMultiplyAcc(dev, Wx_slice, mult_r, dEdxi);
      }
    } else if(i==num_inputs){
      // dh_tm1 = [Wh_i]^T   [di . i_t . (1-i_t)]
      //          [Wh_f]   * [df . f_t . (1-f_t)]
      //          [Wh_o]     [do . o_t . (1-o_t)]
      //          [Wh_g]     [dg . (1 - g_t^2)]
      //    where Wh is broadcasted over batches

      // scratch memory for the matrix multiplication
      Tensor mult_r_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_r_ifo.v = static_cast<float*>(scratch_allocator->allocate(mult_r_ifo.d.size() * sizeof(float)));
      Tensor mult_r_g(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_r_g.v = static_cast<float*>(scratch_allocator->allocate(mult_r_g.d.size() * sizeof(float)));
      Tensor mult_r(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_r.v = static_cast<float*>(scratch_allocator->allocate(mult_r.d.size() * sizeof(float)));

      // mult_r = [di . i_t . (1-i_t)]
      //          [df . f_t . (1-f_t)]
      //          [do . o_t . (1-o_t)]
      //          [dg . (1 - g_t^2)]
      tb<2>(mult_r_ifo).device(*dev.edevice) = tb<2>(dEdf_ifo) * tb<2>(fx_ifo) * (tb<2>(fx_ifo).constant(1) - tb<2>(fx_ifo));
      tb<2>(mult_r_g).device(*dev.edevice) = tb<2>(dEdf_g) * (tb<2>(fx_g).constant(1) - tb<2>(fx_g).square());
      tb<2>(mult_r).slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = tb<2>(mult_r_ifo);
      tb<2>(mult_r).slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = tb<2>(mult_r_g);

      // dx_t += mult_l * mult_r
      if(dropout){
        Tensor mult_y(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
        mult_y.v = static_cast<float*>(scratch_allocator->allocate(mult_y.d.size() * sizeof(float)));
        TensorTools::zero(mult_y);
        MatrixTranspMultiplyAcc(dev, *Wh, mult_r, mult_y);
        tvec(dEdxi).device(*dev.edevice) += tvec(mult_y) * tvec(mask_h);
      } else {
        MatrixTranspMultiplyAcc(dev, *Wh, mult_r, dEdxi);
      }

    } else if(i==num_inputs+1){ // dWx
      // goal: dWx_i = [di . i_t . (1-i_t)] * x_t (here * is outer product), then sum over batches
      //       dWx_f = [di . f_t . (1-f_t)] * x_t (here * is outer product), then sum over batches
      //       dWx_o = [di . o_t . (1-o_t)] * x_t (here * is outer product), then sum over batches
      //       dWx_g = [dg . (1 - g_t^2)] * x_t (here * is outer product), then sum over batches

      // scratch memory for the matrix multiplication
      Tensor mult_l_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l_ifo.v = static_cast<float*>(scratch_allocator->allocate(mult_l_ifo.d.size() * sizeof(float)));
      Tensor mult_l_g(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l_g.v = static_cast<float*>(scratch_allocator->allocate(mult_l_g.d.size() * sizeof(float)));
      Tensor mult_l(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l.v = static_cast<float*>(scratch_allocator->allocate(mult_l.d.size() * sizeof(float)));

      Tensor x_t(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
      if(num_inputs==1){
        x_t.v = xs[0]->v;
      } else {
        Tensor tmp(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
        tmp.v = static_cast<float*>(scratch_allocator->allocate(tmp.d.size() * sizeof(float)));
        Eigen::DSizes<ptrdiff_t, 2> indices_tmp(0, 0);
        Eigen::DSizes<ptrdiff_t, 2> sizes_tmp(0, static_cast<ptrdiff_t>(fx.d.bd));
        for(unsigned i=0; i<num_inputs; i++){
          sizes_tmp[0] = xs[i]->d[0];
          tbvec(tmp).slice(indices_tmp, sizes_tmp).device(*dev.edevice) = tbvec(*xs[i]);
          indices_tmp[0] += xs[i]->d[0];
        }
        x_t.v = tmp.v;
      }


      // mult_l = [di . i_t . (1-i_t)]
      //          [df . f_t . (1-f_t)]
      //          [do . o_t . (1-o_t)]
      //          [dg . (1 - g_t^2)]
      tb<2>(mult_l_ifo).device(*dev.edevice) = tb<2>(dEdf_ifo) * tb<2>(fx_ifo) * (tb<2>(fx_ifo).constant(1) - tb<2>(fx_ifo));
      tb<2>(mult_l_g).device(*dev.edevice) = tb<2>(dEdf_g) * (tb<2>(fx_g).constant(1) - tb<2>(fx_g).square());
      tb<2>(mult_l).slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = tb<2>(mult_l_ifo);
      tb<2>(mult_l).slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = tb<2>(mult_l_g);

      if(dropout){
        Tensor x_t_dropped(Dim({input_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
        x_t_dropped.v = static_cast<float*>(scratch_allocator->allocate(x_t_dropped.d.size() * sizeof(float)));
        tvec(x_t_dropped).device(*dev.edevice) = tvec(x_t) * tvec(mask_x);
        x_t.v = x_t_dropped.v;
      }

      // dWh += (mult_l * mult_r).sum_batches()
      MatrixMultiplyTranspAcc(dev, mult_l, x_t, dEdxi);

    } else if(i==num_inputs+2){ // dWh
      // goal: dWh_i = [di . i_t . (1-i_t)] * h_tm1 (here * is outer product), then sum over batches
      //       dWh_f = [df . f_t . (1-f_t)] * h_tm1 (here * is outer product), then sum over batches
      //       dWh_o = [do . o_t . (1-o_t)] * h_tm1 (here * is outer product), then sum over batches
      //       dWh_g = [dg . (1 - g_t^2)] * h_tm1 (here * is outer product), then sum over batches

      // scratch memory for the matrix multiplication
      Tensor mult_l_ifo(Dim({hidden_dim*3, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l_ifo.v = static_cast<float*>(scratch_allocator->allocate(mult_l_ifo.d.size() * sizeof(float)));
      Tensor mult_l_g(Dim({hidden_dim, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l_g.v = static_cast<float*>(scratch_allocator->allocate(mult_l_g.d.size() * sizeof(float)));
      Tensor mult_l(Dim({hidden_dim*4, 1},batch_size), nullptr, fx.device, fx.mem_pool);
      mult_l.v = static_cast<float*>(scratch_allocator->allocate(mult_l.d.size() * sizeof(float)));

      // mult_l = [di . i_t . (1-i_t)]
      //          [df . f_t . (1-f_t)]
      //          [do . o_t . (1-o_t)]
      //          [dg . (1 - g_t^2)]
      tb<2>(mult_l_ifo).device(*dev.edevice) = tb<2>(dEdf_ifo) * tb<2>(fx_ifo) * (tb<2>(fx_ifo).constant(1) - tb<2>(fx_ifo));
      tb<2>(mult_l_g).device(*dev.edevice) = tb<2>(dEdf_g) * (tb<2>(fx_g).constant(1) - tb<2>(fx_g).square());
      tb<2>(mult_l).slice(indices_mat_i, sizes_mat_3).device(*dev.edevice) = tb<2>(mult_l_ifo);
      tb<2>(mult_l).slice(indices_mat_g, sizes_mat_1).device(*dev.edevice) = tb<2>(mult_l_g);

      if(dropout){
        Tensor h_tm1_dropped(Dim({hidden_dim}, batch_size), nullptr, fx.device, fx.mem_pool);
        h_tm1_dropped.v = static_cast<float*>(scratch_allocator->allocate(h_tm1_dropped.d.size() * sizeof(float)));
        tvec(h_tm1_dropped).device(*dev.edevice) = tvec(h_tm1) * tvec(mask_h);
        h_tm1.v = h_tm1_dropped.v;
      }

      // dWh += (mult_l * mult_r).sum(batches)
      MatrixMultiplyTranspAcc(dev, mult_l, h_tm1, dEdxi);

    } else if(i==num_inputs+3){
      Eigen::DSizes<ptrdiff_t, 1> sizes_1_nobatch(hidden_dim);
      Eigen::DSizes<ptrdiff_t, 1> sizes_3_nobatch(hidden_dim*3);
      Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));
      Eigen::DSizes<ptrdiff_t, 2> sizes_3(hidden_dim*3, static_cast<ptrdiff_t>(fx.d.bd));
      Tensor dEdxi_ifo(Dim({hidden_dim*3},1), nullptr, fx.device, fx.mem_pool);
      dEdxi_ifo.v = static_cast<float*>(scratch_allocator->allocate(dEdxi_ifo.d.size() * sizeof(float)));
      Tensor dEdxi_g(Dim({hidden_dim},1), nullptr, fx.device, fx.mem_pool);
      dEdxi_g.v = static_cast<float*>(scratch_allocator->allocate(dEdxi_g.d.size() * sizeof(float)));

      // db_i = di . i_t . (1-i_t), then sum over batches
      // db_f = df . f_t . (1-f_t), then sum over batches
      // db_o = do . f_t . (1-o_t), then sum over batches
      tvec(dEdxi_ifo).device(*dev.edevice) = (tbvec(dEdf_ifo) * tbvec(fx_ifo) * (tbvec(fx_ifo).constant(1) - tbvec(fx_ifo))).sum(vec_batch_axis);
      tvec(dEdxi).slice(indices_i_nobatch, sizes_3_nobatch).device(*dev.edevice) += tvec(dEdxi_ifo);

      // db_g = dg . (1 - g_t^2), then sum over batches
      tvec(dEdxi_g).device(*dev.edevice) = (tbvec(dEdf_g) * (tbvec(fx_g).constant(1) - tbvec(fx_g).square())).sum(vec_batch_axis);
      tvec(dEdxi).slice(indices_g_nobatch, sizes_1_nobatch).device(*dev.edevice) += tvec(dEdxi_g);
    }
    // no gradients for dropout masks computed

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
    // c_t = i_t . g_t + f_t . c_tm1

    DYNET_ASSERT(xs.size() == 2, "Failed dimension check in VanillaLSTMC::forward");

    const Tensor *c_tm1 = xs[0];
    const Tensor *gates_t = xs[1];

    unsigned hidden_dim = c_tm1->d[0];
    unsigned batch_size = c_tm1->d.bd;

    Eigen::DSizes<ptrdiff_t, 2> indices_i(0, 0);
    Eigen::DSizes<ptrdiff_t, 2> indices_f(hidden_dim,0);
    Eigen::DSizes<ptrdiff_t, 2> indices_g(hidden_dim*3,0);
    Eigen::DSizes<ptrdiff_t, 2> sizes_1(hidden_dim, static_cast<ptrdiff_t>(fx.d.bd));

    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
    Tensor f_t(Dim({hidden_dim,1},batch_size), nullptr, fx.device, fx.mem_pool);
    f_t.v = static_cast<float*>(scratch_allocator->allocate(f_t.d.size() * sizeof(float)));
    tbvec(f_t).device(*dev.edevice) = tbvec(*gates_t).slice(indices_f, sizes_1);

    tbvec(fx).device(*dev.edevice) = tbvec(*gates_t).slice(indices_i, sizes_1);
    tbvec(fx).device(*dev.edevice) = tbvec(fx) * tbvec(*gates_t).slice(indices_g, sizes_1) + tbvec(f_t) * tbvec(*c_tm1);
    scratch_allocator->free();

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
      tbvec(dEdxi).device(*dev.edevice) += tbvec(dEdf) * tbvec(*xs[1]).slice(indices_f, sizes_1);
    } else if(i==1){
      // di_t = dc_t . g_t
      tbvec(dEdxi).slice(indices_i, sizes_1).device(*dev.edevice) += tbvec(dEdf) * tbvec(*xs[1]).slice(indices_g, sizes_1);
      // df_t = dc_t . c_tm1
      tbvec(dEdxi).slice(indices_f, sizes_1).device(*dev.edevice) += tbvec(dEdf) * tbvec(*xs[0]);
      // dg_t = dc_t . i_t
      tbvec(dEdxi).slice(indices_g, sizes_1).device(*dev.edevice) += tbvec(dEdf) * tbvec(*xs[1]).slice(indices_i, sizes_1);
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

    tb<2>(fx).device(*dev.edevice) = tb<2>(*gates_t).slice(indices_o, sizes_1);
    tb<2>(fx).device(*dev.edevice) = tb<2>(fx) * tb<2>(*c_t).tanh();
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

    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];

    if(i==0){
      // dc_t = dh_t . o_t . (1 - tanh^2(c_t)))
      //      = dh_t . o_t . (1 - (h_t cdiv o_t)^2)

      Tensor o_t(Dim({hidden_dim,1},batch_size), nullptr, fx.device, fx.mem_pool);
      o_t.v = static_cast<float*>(scratch_allocator->allocate(o_t.d.size() * sizeof(float)));

      tb<2>(o_t).device(*dev.edevice) = tb<2>(*xs[1]).slice(indices_o, sizes_1);
      tb<2>(dEdxi).device(*dev.edevice) += tb<2>(dEdf)
                                            * tb<2>(o_t)
                                            * (tb<2>(*xs[0]).constant(1) - tb<2>(*xs[0]).tanh().square());
    } else if(i==1){
      Tensor dEdxi_o(Dim({hidden_dim,1},batch_size), nullptr, fx.device, fx.mem_pool);
      dEdxi_o.v = static_cast<float*>(scratch_allocator->allocate(dEdxi_o.d.size() * sizeof(float)));
      // do_t = dh_t . tanh(c_t)
      tb<2>(dEdxi_o).device(*dev.edevice) = tb<2>(dEdf) * tb<2>(*xs[0]).tanh();
      tb<2>(dEdxi).slice(indices_o, sizes_1).device(*dev.edevice) += tb<2>(dEdxi_o);
    }
    scratch_allocator->free();
  }

  DYNET_NODE_INST_DEV_IMPL(VanillaLSTMH)


}
