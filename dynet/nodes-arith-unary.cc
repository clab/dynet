#include "dynet/nodes-arith-unary.h"

#include "dynet/nodes-macros.h"
#include "dynet/functors.h"

using namespace std;

namespace dynet {

// ************* Square *************

#ifndef __CUDACC__

string Square::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "square(" << arg_names[0] << ')';
  return s.str();
}

Dim Square::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Square")
  return xs[0];
}

#endif

template<class MyDevice>
void Square::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().square();
}

template<class MyDevice>
void Square::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * xs[0]->tvec() * 2.f;
}
DYNET_NODE_INST_DEV_IMPL(Square)

// ************* Cube *************

#ifndef __CUDACC__

string Cube::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "cube(" << arg_names[0] << ')';
  return s.str();
}

Dim Cube::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Cube")
  return xs[0];
}

#endif

template<class MyDevice>
void Cube::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().cube();
}

template<class MyDevice>
void Cube::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * xs[0]->tvec().square() * 3.f;
}
DYNET_NODE_INST_DEV_IMPL(Cube)

// ************* Sqrt *************

#ifndef __CUDACC__

string Sqrt::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sqrt(" << arg_names[0] << ')';
  return s.str();
}

Dim Sqrt::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Sqrt")
  return xs[0];
}

#endif

template<class MyDevice>
void Sqrt::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().sqrt();
}

template<class MyDevice>
void Sqrt::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += fx.tvec().binaryExpr(dEdf.tvec(), FSqrtBackward());
}
DYNET_NODE_INST_DEV_IMPL(Sqrt)

// ************* Exp *************

#ifndef __CUDACC__

string Exp::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "exp(" << arg_names[0] << ')';
  return os.str();
}

Dim Exp::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Exp")
  return xs[0];
}

#endif

template<class MyDevice>
void Exp::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().exp();
}

template<class MyDevice>
void Exp::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * fx.tvec();
}
DYNET_NODE_INST_DEV_IMPL(Exp)

// ************* Log *************

#ifndef __CUDACC__

string Log::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "log(" << arg_names[0] << ')';
  return os.str();
}

Dim Log::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Log")
  return xs[0];
}

#endif

template<class MyDevice>
void Log::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().log();
}

template<class MyDevice>
void Log::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() / xs[0]->tvec();
}
DYNET_NODE_INST_DEV_IMPL(Log)

// ************* Negate *************

#ifndef __CUDACC__

string Negate::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << '-' << arg_names[0];
  return s.str();
}

Dim Negate::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Negate");
  return xs[0];
}

#endif

template<class MyDevice>
void Negate::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 1, "Failed dimension check in Negate::forward");
  fx.tvec().device(*dev.edevice) = -xs[0]->tvec();
}

template<class MyDevice>
void Negate::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i == 0, "Failed dimension check in Negate::backward");
  dEdxi.tvec().device(*dev.edevice) -= dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(Negate)

// ************* Abs *************

#ifndef __CUDACC__

string Abs::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "abs(" << arg_names[0] << ')';
  return s.str();
}

Dim Abs::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in Abs")
  return xs[0];
}

#endif

template<class MyDevice>
void Abs::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().abs();
}

template<class MyDevice>
void Abs::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += dEdf.tvec() * xs[0]->tvec().sign();
}
DYNET_NODE_INST_DEV_IMPL(Abs)

// ************* LogGamma *************

#ifndef __CUDACC__

string LogGamma::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "lgamma(" << arg_names[0] << ')';
  return os.str();
}

Dim LogGamma::dim_forward(const vector<Dim>& xs) const {
  DYNET_ARG_CHECK(xs.size() == 1, "Failed input count check in LogGamma")
  return xs[0];
}

#endif

template<class MyDevice>
void LogGamma::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.tvec().device(*dev.edevice) = xs[0]->tvec().lgamma();
}

template<class MyDevice>
void LogGamma::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  dEdxi.tvec().device(*dev.edevice) += xs[0]->tvec().digamma() * dEdf.tvec();
}
DYNET_NODE_INST_DEV_IMPL(LogGamma)

}
