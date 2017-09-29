/**
 * \file param-init.h
 * \defgroup params params
 *
 */

#ifndef DYNET_PARAM_INIT_H_
#define DYNET_PARAM_INIT_H_

#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>

namespace dynet {

struct Tensor;

/**
 * \ingroup params
 * \brief Initializers for parameters
 * \details Allows for custom parameter initialization
 */
struct ParameterInit {
  /**
   * \brief Default constructor
   */
  ParameterInit() {}
  virtual ~ParameterInit() {}
  /**
   * \brief Function called upon initialization
   * \details Whenever you inherit this struct to implement your own custom initializer, this is the function you want to overload to implement your logic.
   *
   * \param values The tensor to be initialized. You should modify it in-place. See dynet/model.cc for some examples
   */
  virtual void initialize_params(Tensor & values) const = 0;
};

/**
 * \ingroup params
 * \brief Initialize parameters with samples from a normal distribution
 */
struct ParameterInitNormal : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param m Mean of the gaussian distribution
   * \param v Variance of the gaussian distribution (reminder : the variance is the __square__ of the standard deviation)
   */
  ParameterInitNormal(float m = 0.0f, float v = 1.0f) : mean(m), var(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float mean, var;
};

/**
 * \ingroup params
 * \brief Initialize parameters with samples from a uniform distribution
 *
 */
struct ParameterInitUniform : public ParameterInit {
  /**
   * \brief Constructor for uniform distribution centered on 0
   * \details [long description]Samples parameters from \f$mathcal U([-\mathrm{scale},+\mathrm{scale}]\f$
   * \param scale Scale of the distribution
   */
  ParameterInitUniform(float scale) :
    left(-scale), right(scale) { if (scale == 0.0f) throw std::domain_error("Scale of the uniform distribution cannot be 0 in ParameterInitUniform"); }
  /**
   * \brief Constructor for uniform distribution in a specific interval
   * \details [long description]
   *
   * \param l Lower bound of the interval
   * \param r Upper bound of the interval
   */
  ParameterInitUniform(float l, float r) : left(l), right(r) { if (l == r) throw std::domain_error("Empty interval in ParameterInitUniform"); }
  virtual void initialize_params(Tensor & values) const override;
private:
  float left, right;
};

/**
 * \ingroup params
 * \brief Initialize parameters with a constant value
 */
struct ParameterInitConst : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param c Constant value
   */
  ParameterInitConst(float c) : cnst(c) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};

/**
 * \ingroup params
 * \brief Initialize as the identity
 * \details This will raise an exception if used on non square matrices
 */
struct ParameterInitIdentity : public ParameterInit {
  /**
   * \brief Constructor
   */
  ParameterInitIdentity() {}
  virtual void initialize_params(Tensor & values) const override;
};

/**
 * \ingroup params
 * \brief Initialize with the methods described in [Glorot, 2010](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf?hc_location=ufi)
 * \details In order to preserve the variance of the forward and backward flow across layers, the parameters \f$\theta\f$ are initialized such that \f$\mathrm{Var}(\theta)=\frac 2 {n_1+n_2}\f$ where \f$n_1,n_2\f$ are the input and output dim.
 * Important note : The underlying distribution is uniform (not gaussian)
 * 
 * *Note:* This is also known as **Xavier initialization**
 *
 */
struct ParameterInitGlorot : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param is_lookup Boolean value identifying the parameter as a LookupParameter
   * \param gain Scaling parameter. In order for the Glorot initialization to be correct, you should ût this equal to \f$\frac 1 {f'(0)}\f$ where \f$f\f$ is your activation function
   */
  ParameterInitGlorot(bool is_lookup = false, float gain = 1.f) : lookup(is_lookup), gain(gain) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  bool lookup;
  float gain;
};
/**
 * \ingroup params
 * \brief Initializes according to [Saxe et al., 2014](https://arxiv.org/abs/1312.6120)
 * \details Initializes as a random orthogonal matrix (unimplemented for GPU)
 */
struct ParameterInitSaxe : public ParameterInit {
  /**
   * \brief Constructor
   */
  ParameterInitSaxe(float gain = 1.0) : gain(gain) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float gain;
};

/**
 * \ingroup params
 * \brief Initializes from a file
 * \details Useful for reusing weights, etc...
 *
 */
struct ParameterInitFromFile : public ParameterInit {
  /**
   * \brief Constructor
   * \param f File name (format should just be a list of values)
   */
  ParameterInitFromFile(std::string f) : filename(f) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::string filename;
};

/**
 * \ingroup params
 * \brief Initializes from a `std::vector` of floats
 */
struct ParameterInitFromVector : public ParameterInit {
  /**
   * \brief Constructor
   *
   * \param v Vector of values to be used
   */
  ParameterInitFromVector(std::vector<float> v) : vec(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::vector<float> vec;
};

}

#endif
