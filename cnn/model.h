#ifndef CNN_PARAMS_H_
#define CNN_PARAMS_H_

#include <vector>
#include <unordered_set>
#include <string>

#include <boost/serialization/export.hpp>

#include "cnn/tensor.h"
#include "cnn/weight-decay.h"

namespace cnn {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

struct ParameterStorageBase {
  friend class Model;
  virtual void scale_parameters(float a) = 0;
  virtual void zero() = 0;
  virtual void squared_l2norm(float* sqnorm) const = 0;
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  virtual size_t size() const = 0;
  virtual ~ParameterStorageBase();
  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive& ar, const unsigned int) {}
};

// represents parameters (e.g., a weight matrix) that will be optimized
struct ParameterStorage : public ParameterStorageBase {
  friend class Model;
  template <class MyDevice>
  void scale_parameters_dev(MyDevice & dev, float a);
  void scale_parameters(float a) override;
  void zero() override;
  template <class MyDevice>
  void squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void squared_l2norm(float* sqnorm) const override;
  template <class MyDevice>
  void g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;

  void copy(const ParameterStorage & val);
  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, const Tensor& g);
  void accumulate_grad(const Tensor& g);
  void clear();

  Dim dim;
  Tensor values;
  Tensor g;
 private:
  ParameterStorage() {}
  explicit ParameterStorage(const Dim& d, float minmax); // initialize with ~U(-minmax,+minmax)
                                 // or Glorot initialization if minmax = 0
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

// represents a matrix/vector embedding of a discrete set
struct LookupParameterStorage : public ParameterStorageBase {
  friend class Model;
  template <class MyDevice>
  void scale_parameters_dev(MyDevice & dev, float a);
  void scale_parameters(float a) override;
  void zero() override;
  template <class MyDevice>
  void squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void squared_l2norm(float* sqnorm) const override;
  template <class MyDevice>
  void g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  template <class MyDevice>
  void initialize_dev(MyDevice & dev, unsigned index, const std::vector<float>& val);
  void initialize(unsigned index, const std::vector<float>& val);

  void copy(const LookupParameterStorage & val);
  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, unsigned index, const Tensor& g);
  void accumulate_grad(unsigned index, const Tensor& g);
  void clear();

  // Initialize each individual lookup from the overall tensors
  void initialize_lookups();

  // Tensors for all dimensions at once
  Dim all_dim;
  Tensor all_values;
  Tensor all_grads;
  // Tensors for each individual lookup
  Dim dim;
  std::vector<Tensor> values;
  std::vector<Tensor> grads;
  // gradients are sparse, so track which components are nonzero
  std::unordered_set<unsigned> non_zero_grads;
 private:
  LookupParameterStorage() {}
  LookupParameterStorage(unsigned n, const Dim& d);
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int) const;
  template<class Archive>
  void load(Archive& ar, const unsigned int);
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

class Model;
struct Parameter {
  Parameter();
  Parameter(const Model* mp, unsigned long index);
  ParameterStorage* get() const;

  // Zero the parameters
  void zero();

  const Model* mp;
  unsigned long index;

  Dim dim() { return get()->dim; }
  Tensor* values() { return &(get()->values); } 

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

struct LookupParameter {
  LookupParameter();
  LookupParameter(const Model* mp, unsigned long index);
  LookupParameterStorage* get() const;
  void initialize(unsigned index, const std::vector<float>& val) const;

  // Zero the parameters
  void zero();

  const Model* mp;
  unsigned long index;

  Dim dim() { return get()->dim; }
  std::vector<Tensor>* values() { return &(get()->values); } 

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

// this is a collection of parameters
// if you need a matrix of parameters, or a lookup table - ask an instance of this class
// this knows how to serialize itself
// parameters know how to track their gradients, but any extra information (like velocity) will live here
class Model {
 public:
  Model();
  ~Model();
  template <class MyDevice>
  float gradient_l2_norm_dev(MyDevice & dev) const;
  float gradient_l2_norm() const;
  void reset_gradient();
  // set scale to use custom initialization
  Parameter add_parameters(const Dim& d, float scale = 0.0f);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d);
  // project weights so their L2 norm = radius
  void project_weights(float radius = 1.0f);
  void set_weight_decay_lambda(float lambda);

  const std::vector<ParameterStorageBase*>& all_parameters_list() const { return all_params; }
  const std::vector<ParameterStorage*>& parameters_list() const { return params; }
  const std::vector<LookupParameterStorage*>& lookup_parameters_list() const { return lookup_params; }

  // Returns the total number of tunable parameters (i. e. scalars) contained within this model.
  // That is to say, a 2x2 matrix counts as four parameters.
  size_t parameter_count() const;

  L2WeightDecay weight_decay;
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);

  std::vector<ParameterStorageBase*> all_params;
  std::vector<ParameterStorage*> params;
  std::vector<LookupParameterStorage*> lookup_params;
  mutable float* gradient_norm_scratch;
};

void save_cnn_model(std::string filename, Model* model);
void load_cnn_model(std::string filename, Model* model);

} // namespace cnn

BOOST_CLASS_EXPORT_KEY(cnn::ParameterStorage)
BOOST_CLASS_EXPORT_KEY(cnn::LookupParameterStorage)

#endif
