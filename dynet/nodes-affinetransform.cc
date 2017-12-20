#include "dynet/nodes-affinetransform.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/matrix-multiply.h"
#include "dynet/tensor-eigen.h"

using namespace std;

namespace dynet {

// ************* AffineTransform *************

#ifndef __CUDACC__

string AffineTransform::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

Dim AffineTransform::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK((xs.size() - 1) % 2 == 0, "Bad number of inputs in AffineTransform: " << xs);
  if(xs.size() == 1) return xs[0];
  DYNET_ARG_CHECK(xs[0].rows() == xs[1].rows() && xs[1].cols() == xs[2].rows(),
                          "Bad dimensions for AffineTransform: " << xs);
  Dim d = (xs[2].cols() != 1 ?
           Dim({xs[0].rows(), xs[2].cols()}, max(max(xs[0].bd, xs[1].bd), xs[2].bd)) :
           Dim({xs[0].rows()}, max(max(xs[0].bd, xs[1].bd), xs[2].bd)));
  for (unsigned i = 3; i < xs.size(); i += 2) {
    DYNET_ARG_CHECK(xs[i].cols() == xs[i+1].rows() && d.rows() == xs[i].rows() && d.cols() == xs[i+1].cols(),
                            "Bad dimensions for AffineTransform: " << xs);
    d.bd = max(max(d.bd, xs[i].bd), xs[i+1].bd);
  }
  return d;
}

int AffineTransform::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::affine);
  // This is a heuristic: we assume that we often have "b + W * x" shaped affine transforms
  // so when everything is batch size one, optimize for this case
  if(dim.bd == 1) {
    s.add_node(args[0]);
    for(size_t i = 1; i < args.size(); i += 2) {
      s.add_node(args[i]);
      s.add_dim(cg.nodes[args[i+1]]->dim); // TODO: this is not the exact same as dim->print_profile
    }
  } else {
    for(auto nid : args) {
      const Dim & d = cg.nodes[nid]->dim;
      if(d.bd == 1)
        s.add_node(nid);
      else
        s.add_dim(d); // TODO: this is not the exact same as dim->print_profile
    }
  }
  return sm.get_idx(s);
}

std::vector<int> AffineTransform::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 0);
  if(dim.bd == 1) {
    for(size_t i = 2; i < ret.size(); i += 2)
      ret[i] = 1;
  } else {
    for(size_t i = 0; i < ret.size(); ++i)
      ret[i] = (cg.nodes[args[i]]->dim.bd > 1);
  }
  return ret;
}

#endif

// Affine transform uses different implementations for CPU and GPU because this is
// much faster than using Eigen's tensor contractions (as of the writing)
template<class MyDevice>
void AffineTransform::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() % 2 == 1, "Failed dimension check in AffineTransform::forward");
  if (xs.size() == 1) {
    tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
    return;
  } else {
    // Add the first matrix
    size_t b_size = xs[0]->d.size(), fx_size = fx.d.size();
    if(fx_size == b_size) {
      tvec(fx).device(*dev.edevice) = tvec(*xs[0]);
    } else {
#ifdef __CUDACC__
      Eigen::array<int, 3> bcast; bcast[0] = 1; bcast[1] = fx.d[1]/xs[0]->d[1]; bcast[2] = fx.d.bd/xs[0]->d.bd;
      tb<2>(fx).device(*dev.edevice) = tb<2>(*xs[0]).broadcast(bcast);
#else
      DYNET_ARG_CHECK(xs[0]->d.bd == 1, "In AffineTransform, broadcasting over columns with mini-batched inputs is not implemented yet");
      float *curr_ptr = fx.v, *end_ptr = curr_ptr + fx.d.size(), *in_ptr = xs[0]->v;
      do {
        memcpy(curr_ptr, in_ptr, sizeof(float)*b_size);
        curr_ptr += b_size;
      } while(curr_ptr != end_ptr);
#endif
    }

    // Perform multiplication
#ifdef __CUDACC__
    for (unsigned i = 1; i < xs.size(); i += 2)
      // fx = (acc_sclar)*fx + xs[0] * xs[1]
      MatrixMultiply(dev, *xs[i], *xs[i + 1], fx, dev.kSCALAR_ONE);
#else
    // Multiply
    for (unsigned i = 1; i < xs.size(); i += 2) {
      if(xs[i]->d.bd == 1 && xs[i+1]->d.bd == fx.d.bd) {
        colbatch_matrix(fx).noalias() += mat(*xs[i]) * colbatch_matrix(*xs[i+1]);
      } else {
        DYNET_ASSERT(xs[i+1]->d.bd == 1 || xs[i+1]->d.bd == xs[i]->d.bd, "Failed dimension check in AffineTransform::forward");
        for(unsigned b = 0; b < fx.d.bd; ++b) {
          batch_matrix(fx, b).noalias() += batch_matrix(*xs[i], b) * batch_matrix(*xs[i+1], b);
        }
      }
    }
#endif
  }
}

template<class MyDevice>
void AffineTransform::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < xs.size(), "Failed boundary check in AffineTransform::backward");
  // Bias term
  if (i == 0) { // bias term
    size_t dx_size = dEdxi.d.size(), df_size = dEdf.d.size();
    if(dx_size == df_size) {
      tvec(dEdxi).device(*dev.edevice) += tvec(dEdf);
    } else {
      DYNET_ARG_CHECK(dEdxi.d.bd == 1, "In AffineTransform, broadcasting over columns with mini-batched inputs is not implemented yet");
#ifdef __CUDACC__
      if(dEdxi.d[1] == dEdf.d[1]) {
        Eigen::array<int, 1> red_axis; red_axis[0] = 2;
        t<2>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).sum(red_axis);
      } else {
        Eigen::array<int, 2> red_axis; red_axis[0] = 1; red_axis[1] = 2;
        t<1>(dEdxi).device(*dev.edevice) += tb<2>(dEdf).sum(red_axis);
      }
#else
      if(dEdxi.d[1] == dEdf.d[1]) {
        for(unsigned b = 0; b < dEdf.d.bd; ++b)
          mat(dEdxi).noalias() += batch_matrix(dEdf, b);
      } else {
        Tensor mychip(dEdxi.d, dEdf.v, dEdf.device, dEdf.mem_pool);
        size_t len = dEdf.d.bd * dEdf.d[1];
        for(unsigned b = 0; b < len; ++b) {
          mat(dEdxi).noalias() += mat(mychip);
          mychip.v += dx_size;
        }
      }
#endif
    }

  // Left argument of matrix multiply
  } else if (i % 2 == 1) {
    int max_b = max(dEdf.d.bd, xs[i+1]->d.bd);
#ifdef __CUDACC__
    if(dEdxi.d.bd == 1 && (dEdf.d.bd == xs[i+1]->d.bd)) {
      CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols() * dEdf.d.batch_elems(),
            dev.kSCALAR_ONE,
            dEdf.v, dEdf.d.rows(),
            xs[i+1]->v, xs[i+1]->d.rows(),
            dev.kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      for(int b = 0; b < max_b; ++b)
        CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
              dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
              dev.kSCALAR_ONE,
              dEdf.batch_ptr(b), dEdf.d.rows(),
              xs[i+1]->batch_ptr(b), xs[i+1]->d.rows(),
              dev.kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
    }
#else
    if(dEdxi.d.bd == 1 && (dEdf.d.bd == xs[i+1]->d.bd)) {
      mat(dEdxi).noalias() += colbatch_matrix(dEdf) * colbatch_matrix(*xs[i+1]).transpose();
    } else {
      for(int b = 0; b < max_b; ++b)
        batch_matrix(dEdxi, b).noalias() += batch_matrix(dEdf, b) * batch_matrix(*xs[i+1], b).transpose();
    }
#endif
  } else {  // right argument of matrix multiply
    int max_b = max(xs[i-1]->d.bd, dEdf.d.bd);
#ifdef __CUDACC__
    // Do a single multiply if xs[i-1] has one batch
    if(xs[i-1]->d.bd == 1 && dEdxi.d.bd == dEdf.d.bd) {
      CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dEdxi.d.rows(), dEdxi.d.cols()*dEdxi.d.batch_elems(), xs[i-1]->d.rows(),
            dev.kSCALAR_ONE,
            xs[i-1]->v, xs[i-1]->d.rows(),
            dEdf.v, dEdf.d.rows(),
            dev.kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      for(int b = 0; b < max_b; ++b)
        CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              dEdxi.d.rows(), dEdxi.d.cols(), xs[i-1]->d.rows(),
              dev.kSCALAR_ONE,
              xs[i-1]->batch_ptr(b), xs[i-1]->d.rows(),
              dEdf.batch_ptr(b), dEdf.d.rows(),
              dev.kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
    }
#else
    if(xs[i-1]->d.bd == 1 && dEdxi.d.bd == dEdf.d.bd) {
      colbatch_matrix(dEdxi).noalias() += mat(*xs[i-1]).transpose() * colbatch_matrix(dEdf);
    } else {
      for(int b = 0; b < max_b; ++b)
        batch_matrix(dEdxi, b).noalias() += batch_matrix(*xs[i-1], b).transpose() * batch_matrix(dEdf, b);
    }
#endif
  }
}
DYNET_NODE_INST_DEV_IMPL(AffineTransform)

}
