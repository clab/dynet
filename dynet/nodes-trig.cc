#include "dynet/tensor-eigen.h"
#include "dynet/nodes-trig.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/simd-functors.h"

using namespace std;

namespace dynet {

// ************* Sin *************

#ifndef __CUDACC__

string Sin::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sin(" << arg_names[0] << ')';
  return s.str();
}

Dim Sin::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Sin")
  return xs[0];
}

#endif

template<class MyDevice>
void Sin::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_sin_op<float>());
}

template<class MyDevice>
void Sin::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_cos_op<float>()) *
      tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Sin)

// ************* Cos *************

#ifndef __CUDACC__

string Cos::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "cos(" << arg_names[0] << ')';
  return s.str();
}

Dim Cos::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Sin")
  return xs[0];
}

#endif

template<class MyDevice>
void Cos::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_cos_op<float>());
}

template<class MyDevice>
void Cos::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) -=
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_sin_op<float>()) *
      tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Cos)

// ************* Tan *************

#ifndef __CUDACC__

string Tan::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tan(" << arg_names[0] << ')';
  return s.str();
}

Dim Tan::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Tan")
  return xs[0];
}

#endif

template<class MyDevice>
void Tan::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_tan_op<float>());
}

template<class MyDevice>
void Tan::backward_dev_impl(const MyDevice & dev,
                            const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(fx).binaryExpr(tvec(dEdf), scalar_tan_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Tan)

// ************* Asin *************

#ifndef __CUDACC__

string Asin::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "asin(" << arg_names[0] << ')';
  return s.str();
}

Dim Asin::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Asin")
  return xs[0];
}

#endif

template<class MyDevice>
void Asin::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_asin_op<float>());
}

template<class MyDevice>
void Asin::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_asin_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Asin)

// ************* Acos *************

#ifndef __CUDACC__

string Acos::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "acos(" << arg_names[0] << ')';
  return s.str();
}

Dim Acos::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Acos")
  return xs[0];
}

#endif

template<class MyDevice>
void Acos::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_acos_op<float>());
}

template<class MyDevice>
void Acos::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_acos_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Acos)

// ************* Atan *************

#ifndef __CUDACC__

string Atan::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "atan(" << arg_names[0] << ')';
  return s.str();
}

Dim Atan::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Atan")
  return xs[0];
}

#endif

template<class MyDevice>
void Atan::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_atan_op<float>());
}

template<class MyDevice>
void Atan::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_atan_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Atan)

// ************* Sinh *************

#ifndef __CUDACC__

string Sinh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sinh(" << arg_names[0] << ')';
  return s.str();
}

Dim Sinh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Sinh")
  return xs[0];
}

#endif

template<class MyDevice>
void Sinh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_sinh_op<float>());
}

template<class MyDevice>
void Sinh::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_cosh_op<float>()) *
      tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Sinh)

// ************* Cosh *************

#ifndef __CUDACC__

string Cosh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "cosh(" << arg_names[0] << ')';
  return s.str();
}

Dim Cosh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Cosh")
  return xs[0];
}

#endif

template<class MyDevice>
void Cosh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_cosh_op<float>());
}

template<class MyDevice>
void Cosh::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).unaryExpr(Eigen::internal::scalar_sinh_op<float>()) *
      tvec(dEdf);
}
DYNET_NODE_INST_DEV_IMPL(Cosh)

// ************* Tanh *************

#ifndef __CUDACC__

string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

Dim Tanh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Tanh")
  return xs[0];
}

#endif

template<class MyDevice>
void Tanh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) = tvec(*xs[0]).tanh();
}

template<class MyDevice>
void Tanh::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) += 
      tvec(fx).binaryExpr(tvec(dEdf), scalar_tanh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Tanh)

// ************* Asinh *************

#ifndef __CUDACC__

string Asinh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "asinh(" << arg_names[0] << ')';
  return s.str();
}

Dim Asinh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Asinh")
  return xs[0];
}

#endif

template<class MyDevice>
void Asinh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(scalar_asinh_forward_op<float>());
}

template<class MyDevice>
void Asinh::backward_dev_impl(const MyDevice & dev,
                              const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_asinh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Asinh)

// ************* Acosh *************

#ifndef __CUDACC__

string Acosh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "acosh(" << arg_names[0] << ')';
  return s.str();
}

Dim Acosh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Acosh")
  return xs[0];
}

#endif

template<class MyDevice>
void Acosh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(scalar_acosh_forward_op<float>());
}

template<class MyDevice>
void Acosh::backward_dev_impl(const MyDevice & dev,
                              const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_acosh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Acosh)

// ************* Atanh *************

#ifndef __CUDACC__

string Atanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "atanh(" << arg_names[0] << ')';
  return s.str();
}

Dim Atanh::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Atanh")
  return xs[0];
}

#endif

template<class MyDevice>
void Atanh::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  tvec(fx).device(*dev.edevice) =
      tvec(*xs[0]).unaryExpr(scalar_atanh_forward_op<float>());
}

template<class MyDevice>
void Atanh::backward_dev_impl(const MyDevice & dev,
                              const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  tvec(dEdxi).device(*dev.edevice) +=
      tvec(*xs[0]).binaryExpr(tvec(dEdf), scalar_atanh_backward_op<float>());
}
DYNET_NODE_INST_DEV_IMPL(Atanh)

}  // namespace dynet
