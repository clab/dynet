#ifndef DYNET_XFUNCTORS_H
#define DYNET_XFUNCTORS_H

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include "dynet/functors.h"

// these functors are implemented to exploit Eigen's internal logic for doing
// vectorized arithmetic. I'm putting them in a separate file since, if Eigen
// breaks backward compatibility by changing an internal interface, I want
// the necessary changes to be localized.
//
// to implement your own functor, you need to provide
//   1) operator() implemented on the scalar data type
//   2) packetOp implemented using vector ("packet") type
//   3) the functor_traits specialization for your functor
//      that tells the compiler whether your architecture
//      has vectorized support for the operations you need
//      and an estimate of the cost of the operation

namespace dynet {
template<typename Scalar> struct const_add_op {
  const_add_op(const Scalar& c) : c(c) {}
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    return c + x;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    return padd(pset1<Packet>(c), x);
  }
  Scalar c;
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::const_add_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2,
    PacketAccess = packet_traits<Scalar>::HasAdd
  };
};
} }

namespace dynet {
template<typename Scalar> struct const_minus_op {
  const_minus_op(const Scalar& c) : c(c) {}
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    return c - x;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    return psub(pset1<Packet>(c), x);
  }
  Scalar c;
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::const_minus_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2,
    PacketAccess = packet_traits<Scalar>::HasSub
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_logistic_sigmoid_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_logistic_sigmoid_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    const float one = 1.0f;
    return one / (one + Eigen::numext::exp(-x));
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& _x) const {
    // This implementation is copied from Eigen
    // See https://github.com/eigenteam/eigen-git-mirror/blob/1e1794c6020e54c932918192ac39285f8ed2d828/Eigen/src/Core/functors/UnaryFunctors.h#L916

    // Clamp the inputs to the range [-18, 18] since anything outside
    // this range is 0.0f or 1.0f in single-precision.
    using namespace Eigen::internal;
    const Packet x = pmax(pmin(_x, pset1<Packet>(18.0)), pset1<Packet>(-18.0));

    // The monomial coefficients of the numerator polynomial (odd).
    const Packet alpha_1 = pset1<Packet>(2.48287947061529e-01);
    const Packet alpha_3 = pset1<Packet>(8.51377133304701e-03);
    const Packet alpha_5 = pset1<Packet>(6.08574864600143e-05);
    const Packet alpha_7 = pset1<Packet>(1.15627324459942e-07);
    const Packet alpha_9 = pset1<Packet>(4.37031012579801e-11);

    // The monomial coefficients of the denominator polynomial (even).
    const Packet beta_0 = pset1<Packet>(9.93151921023180e-01);
    const Packet beta_2 = pset1<Packet>(1.16817656904453e-01);
    const Packet beta_4 = pset1<Packet>(1.70198817374094e-03);
    const Packet beta_6 = pset1<Packet>(6.29106785017040e-06);
    const Packet beta_8 = pset1<Packet>(5.76102136993427e-09);
    const Packet beta_10 = pset1<Packet>(6.10247389755681e-13);

    // Since the polynomials are odd/even, we need x^2.
    const Packet x2 = pmul(x, x);

    // Evaluate the numerator polynomial p.
    Packet p = pmadd(x2, alpha_9, alpha_7);
    p = pmadd(x2, p, alpha_5);
    p = pmadd(x2, p, alpha_3);
    p = pmadd(x2, p, alpha_1);
    p = pmul(x, p);

    // Evaluate the denominator polynomial p.
    Packet q = pmadd(x2, beta_10, beta_8);
    q = pmadd(x2, q, beta_6);
    q = pmadd(x2, q, beta_4);
    q = pmadd(x2, q, beta_2);
    q = pmadd(x2, q, beta_0);

    // Divide the numerator by the denominator and shift it up.
    return pmax(pmin(padd(pdiv(p, q), pset1<Packet>(0.5)), pset1<Packet>(1.0)),
                pset1<Packet>(0.0));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_logistic_sigmoid_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 6,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasDiv &&
                   packet_traits<Scalar>::HasNegate && packet_traits<Scalar>::HasExp
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_log_sigmoid_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log_sigmoid_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    using std::log1pf;
    // distinguish between positive and negative values of x for precision
    if (x>0)
        return -log1pf(expf(-x));
    else
        return x - log1pf(expf(x));
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    const Packet minus_one = pset1<Packet>(-1.0);
    // Trick to mimick a condition do the computation for both cases and take the min/max with a "pivot" value (here -1) then add. Then substract the excess -1
    return pmin(
            padd(
             // Negative case (close to x)
             pmin(
                 minus_one,
                 psub(x, plog1p(pexp(x)))
                 ),
             // Positive case (close to 0)
             pmax(
                 minus_one,
                 pnegate(plog1p(pexp(pnegate(x))))
                 )
             ),
            minus_one);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_log_sigmoid_forward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost *6 + NumTraits<Scalar>::MulCost * 4,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasSub && 
                   packet_traits<Scalar>::HasMin && packet_traits<Scalar>::HasMax && 
                   packet_traits<Scalar>::HasLog1p && packet_traits<Scalar>::HasExp &&
                   packet_traits<Scalar>::HasNegate
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_log_sigmoid_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log_sigmoid_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { 
    return (1 - expf(t)) * d;
  }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, pexp(t)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_log_sigmoid_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasExp
  };
};
}}

namespace dynet {
template<typename Scalar> struct scalar_sqrt_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& t, const Scalar& d) const {
    const Scalar two = Scalar(2);
    return d / (two * t);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet two = pset1<Packet>(2);
    return pdiv(d, pmul(two, t));
  }
};
typedef scalar_sqrt_backward_op<float> FSqrtBackward;
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_sqrt_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::MulCost * 2,
    PacketAccess = packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasDiv
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_asinh_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asinh_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
#ifndef __CUDACC__
    return std::asinh(x);
#else
    return ::asinh(x);
#endif
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
#ifndef __CUDACC__
    return std::asinh(x);
#else
    return ::asinh(x);
#endif
  }
};
}

namespace dynet {
template<typename Scalar> struct scalar_acosh_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acosh_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
#ifndef __CUDACC__
    return std::acosh(x);
#else
    return ::acosh(x);
#endif
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
#ifndef __CUDACC__
    return std::acosh(x);
#else
    return ::acosh(x);
#endif
  }
};
}

namespace dynet {
template<typename Scalar> struct scalar_atanh_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atanh_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
#ifndef __CUDACC__
    return std::atanh(x);
#else
    return ::atanh(x);
#endif
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
#ifndef __CUDACC__
    return std::atanh(x);
#else
    return ::atanh(x);
#endif
  }
};
}

namespace dynet {
template<typename Scalar> struct scalar_tan_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tan_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { return (1 + t * t) * d; }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(pmadd(t, t, one), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_tan_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul
  };
};
}}

namespace dynet {
template<typename Scalar> struct scalar_asin_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asin_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / sqrt(1 - x * x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(prsqrt(psub(one, pmul(x, x))), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_asin_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasNegate && packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_acos_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acos_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return -d / sqrt(1 - x * x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pnegate(pmul(prsqrt(psub(one, pmul(x, x))), d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_acos_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasNegate && packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_atan_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atan_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / (x * x + 1);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    //return pdiv(d, padd(pmul(x, x), one));
    return pdiv(d, pmadd(x, x, one));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_atan_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasDiv
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_asinh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asinh_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / sqrt(x * x + 1);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(prsqrt(pmadd(x, x, one)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_asinh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_acosh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acosh_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / sqrt(x * x - 1);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(prsqrt(psub(pmul(x, x), one)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_acosh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_atanh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atanh_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / (1 - x * x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pdiv(d, psub(one, pmul(x, x)));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_atanh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 3,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasDiv
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_erf_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erf_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    const Scalar sqrt_pi_over2(1.1283791670955125738961589);
    return sqrt_pi_over2 * expf(-x * x) * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet sqrt_pi_over2 = pset1<Packet>(1.1283791670955125738961589);
    return pmul(sqrt_pi_over2, pmul(pexp(pnegate(pmul(x, x))), d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_erf_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::MulCost * 8,
    PacketAccess = packet_traits<Scalar>::HasExp && packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasNegate
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_logistic_sigmoid_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_logistic_sigmoid_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& t, const Scalar& d) const {
    const Scalar one = Scalar(1);
    return (one - t) * t * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, t), pmul(t, d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_logistic_sigmoid_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + NumTraits<Scalar>::MulCost * 2,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_tanh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { return (1 - t * t) * d; }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, pmul(t, t)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_tanh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul
  };
};
}}

namespace dynet {
//this is slower than the dumb implementation, probably because of the pset operations
// which could be factored out into the constructor, but the Packet type isn't used
// then (and I think fixing this would be hard)
template<typename Scalar> struct scalar_nlsoftmax_backward_op {
  scalar_nlsoftmax_backward_op(const Scalar& lz, const Scalar& err) : logz(lz), d(err) {}
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& t) const {
    return expf(t - logz) * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t) const {
    using namespace Eigen::internal;
    const Packet lz = pset1<Packet>(logz);
    const Packet dd = pset1<Packet>(d);
    return pmul(pexp(psub(t, lz)), dd);
  }
  Scalar logz;
  Scalar d;
};}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_nlsoftmax_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 6 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasExp
  };
};
}}

#endif
