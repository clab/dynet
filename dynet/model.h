#ifndef DYNET_PARAMS_H_
#define DYNET_PARAMS_H_

#include <vector>
#include <set>
#include <unordered_set>
#include <string>

#include <boost/serialization/export.hpp>

#include "dynet/tensor.h"
#include "dynet/weight-decay.h"

namespace dynet {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

struct ParameterInit;

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
  void serialize(Archive& /* ar */, const unsigned int) {}
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
  explicit ParameterStorage(const Dim& d, const ParameterInit & init); // initialize with custom initializer
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
  LookupParameterStorage(unsigned n, const Dim& d, const ParameterInit & init);
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
  Parameter(Model* mp, unsigned long index);
  ParameterStorage* get() const;

  // Zero the parameters
  void zero();

  Model* mp;
  unsigned long index;

  Dim dim() { return get()->dim; }
  Tensor* values() { return &(get()->values); } 

  void set_updated(bool b);
  bool is_updated();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

struct LookupParameter {
  LookupParameter();
  LookupParameter(Model* mp, unsigned long index);
  LookupParameterStorage* get() const;
  void initialize(unsigned index, const std::vector<float>& val) const;

  // Zero the parameters
  void zero();

  Model* mp;
  unsigned long index;

  Dim dim() { return get()->dim; }
  std::vector<Tensor>* values() { return &(get()->values); } 

  void set_updated(bool b);
  bool is_updated();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);
};

// Initilizers for parameters
struct ParameterInit {
  ParameterInit() {}
  virtual ~ParameterInit() {}
  virtual void initialize_params(Tensor & values) const = 0;
};

struct ParameterInitNormal : public ParameterInit {
  ParameterInitNormal(float m = 0.0f, float v = 1.0f) : mean(m), var(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float mean, var;
};

struct ParameterInitUniform : public ParameterInit {
  ParameterInitUniform(float scale) :
    left(-scale), right(scale) { assert(scale != 0.0f); }
  ParameterInitUniform(float l, float r) : left(l), right(r) { assert(l != r); }
  virtual void initialize_params(Tensor & values) const override;
private:
  float left, right;
};

struct ParameterInitConst : public ParameterInit {
  ParameterInitConst(float c) : cnst(c) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};

struct ParameterInitIdentity : public ParameterInit {
  ParameterInitIdentity() {}
  virtual void initialize_params(Tensor & values) const override;
};

struct ParameterInitGlorot : public ParameterInit {
  ParameterInitGlorot(bool is_lookup = false) : lookup(is_lookup) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  bool lookup;
};

struct ParameterInitSaxe : public ParameterInit {
  ParameterInitSaxe() {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};

struct ParameterInitFromFile : public ParameterInit {
  ParameterInitFromFile(std::string f) : filename(f) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::string filename;
};

struct ParameterInitFromVector : public ParameterInit {
  ParameterInitFromVector(std::vector<float> v) : vec(v) {}
  virtual void initialize_params(Tensor & values) const override;
private:
  std::vector<float> vec;
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
  Parameter add_parameters(const Dim& d, const ParameterInit & init);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init);
  // project weights so their L2 norm = radius
  void project_weights(float radius = 1.0f);
  void set_weight_decay_lambda(float lambda);

  //const std::vector<ParameterStorageBase*>& all_parameters_list() const { return all_params; }
  const std::vector<ParameterStorage*>& parameters_list() const { return params; }
  const std::vector<LookupParameterStorage*>& lookup_parameters_list() const { return lookup_params; }

  // indexes into params and lookup_params
  const std::vector<unsigned>& updated_parameters_list() const { return updated_params; }
  const std::vector<unsigned>& updated_lookup_parameters_list() const { return updated_lookup_params; }

  // Returns the total number of tunable parameters (i. e. scalars) contained within this model.
  // That is to say, a 2x2 matrix counts as four parameters.
  size_t parameter_count() const;
  size_t updated_parameter_count() const;

  void set_updated_param(const Parameter *p, bool status);
  void set_updated_lookup_param(const LookupParameter *p, bool status);
  bool is_updated_param(const Parameter *p);
  bool is_updated_lookup_param(const LookupParameter *p);

  L2WeightDecay weight_decay;
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int);

  std::vector<ParameterStorageBase*> all_params;
  std::vector<ParameterStorage*> params;
  std::vector<LookupParameterStorage*> lookup_params;

  // these are a subset of the parameters that are used when model is updated.
  // kept as indices into params and lookup_params.
  std::vector<unsigned> updated_params;
  std::vector<unsigned> updated_lookup_params;

  mutable float* gradient_norm_scratch;
};

void save_dynet_model(std::string filename, Model* model);
void load_dynet_model(std::string filename, Model* model);

} // namespace dynet

BOOST_CLASS_EXPORT_KEY(dynet::ParameterStorage)
BOOST_CLASS_EXPORT_KEY(dynet::LookupParameterStorage)

#endif
