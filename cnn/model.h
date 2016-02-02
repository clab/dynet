#ifndef CNN_PARAMS_H_
#define CNN_PARAMS_H_

#include <vector>
#include <unordered_set>
#include <string>

#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/tensor.h"

namespace cnn {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

struct ParametersBase {
  friend class Model;
  virtual void scale_parameters(float a) = 0;
  virtual void squared_l2norm(float* sqnorm) const = 0;
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  virtual size_t size() const = 0;
  virtual ~ParametersBase();
  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive& ar, const unsigned int) {}
};

// represents parameters (e.g., a weight matrix) that will be optimized
struct Parameters : public ParametersBase {
  friend class Model;
  void scale_parameters(float a) override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;

  void copy(const Parameters & val);
  void accumulate_grad(const Tensor& g);
  void clear();

  Dim dim;
  Tensor values;
  Tensor g;
 private:
  Parameters() {}
  explicit Parameters(const Dim& d, float minmax); // initialize with ~U(-minmax,+minmax)
                                 // or Glorot initialization if minmax = 0
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::base_object<ParametersBase>(*this);
    ar & dim;
    ar & values;
    ar & g;
  }
};

// represents a matrix/vector embedding of a discrete set
struct LookupParameters : public ParametersBase {
  friend class Model;
  void scale_parameters(float a) override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  void Initialize(unsigned index, const std::vector<float>& val);

  void copy(const LookupParameters & val);
  void accumulate_grad(unsigned index, const Tensor& g);
  void clear();

  Dim dim;
  std::vector<Tensor> values;
  std::vector<Tensor> grads;
  // gradients are sparse, so track which components are nonzero
  std::unordered_set<unsigned> non_zero_grads;
 private:
  LookupParameters() {}
  LookupParameters(unsigned n, const Dim& d);
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    boost::serialization::base_object<ParametersBase>(*this);
    ar & dim;
    ar & values;
    ar & grads;
  }
};

class Model;
struct ParameterIndex {
  ParameterIndex();
  ParameterIndex(const Model* mp, unsigned long index);
  Parameters* get() const;
  Dim& dim() const;
  Tensor& values() const;
  void accumulate_grad(const Tensor& g) const;
  void scale_parameters(float a);

  const Model* mp;
  unsigned long index;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & mp;
    ar & index;
  }
};

struct LookupParameterIndex {
  LookupParameterIndex();
  LookupParameterIndex(const Model* mp, unsigned long index);
  LookupParameters* get() const;
  Dim& dim() const;
  std::vector<Tensor>& values() const;
  void accumulate_grad(unsigned index, const Tensor& g) const;
  void Initialize(unsigned index, const std::vector<float>& val) const;
  void scale_parameters(float a);

  const Model* mp;
  unsigned long index;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & mp;
    ar & index;
  }
};

// this is a collection of parameters
// if you need a matrix of parameters, or a lookup table - ask an instance of this class
// this knows how to serialize itself
// parameters know how to track their gradients, but any extra information (like velocity) will live here
class Model {
 public:
  Model() : gradient_norm_scratch() {}
  ~Model();
  float gradient_l2_norm() const;
  void reset_gradient();
  // set scale to use custom initialization
  ParameterIndex add_parameters(const Dim& d, float scale = 0.0f);
  LookupParameterIndex add_lookup_parameters(unsigned n, const Dim& d);
  // project weights so their L2 norm = radius
  void project_weights(float radius = 1.0f);

  const std::vector<ParametersBase*>& all_parameters_list() const { return all_params; }
  const std::vector<Parameters*>& parameters_list() const { return params; }
  const std::vector<LookupParameters*>& lookup_parameters_list() const { return lookup_params; }

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & all_params;
    ar & params;
    ar & lookup_params;
  }

  std::vector<ParametersBase*> all_params;
  std::vector<Parameters*> params;
  std::vector<LookupParameters*> lookup_params;
  mutable float* gradient_norm_scratch;
};

void save_cnn_model(std::string filename, Model* model);
void load_cnn_model(std::string filename, Model* model);

} // namespace cnn
BOOST_CLASS_EXPORT_KEY(cnn::Parameters)
BOOST_CLASS_EXPORT_KEY(cnn::LookupParameters)

#endif
