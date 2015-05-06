#ifndef CNN_PARAMS_H_
#define CNN_PARAMS_H_

#include <vector>
#include <unordered_set>

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

#include "cnn/tensor.h"

namespace cnn {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

struct ParametersBase {
  friend class Model;
  virtual void rescale_gradient(real scale) = 0;
  virtual real g_squared_l2norm() const = 0;
  virtual size_t size() const = 0;
  virtual ~ParametersBase();
};

// represents parameters (e.g., a weight matrix) that will be optimized
struct Parameters : public ParametersBase {
  friend class Model;
  void rescale_gradient(real scale) override;
  real g_squared_l2norm() const override;
  size_t size() const override;

  void accumulate_grad(const Tensor& g);
  void clear();

  Dim dim;
  Tensor values;
  Tensor g;
 private:
  Parameters() {}
  explicit Parameters(const Dim& d);
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & dim;
    ar & values;
  }
};

// represents a matrix/vector embedding of a discrete set
struct LookupParameters : public ParametersBase {
  friend class Model;
  void rescale_gradient(real scale) override;
  real g_squared_l2norm() const override;
  size_t size() const override;
  void Initialize(unsigned index, const std::vector<float>& val);

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
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & dim;
    ar & values;
  }
};

// this is a collection of parameters
// if you need a matrix of parameters, or a lookup table - ask an instance of this class
// this knows how to serialize itself
// parameters know how to track their gradients, but any extra information (like velocity) will live here
class Model {
 public:
  ~Model();
  Parameters* add_parameters(const Dim& d);  // initialized randomly
  LookupParameters* add_lookup_parameters(unsigned n, const Dim& d);

  const std::vector<ParametersBase*>& all_parameters_list() const { return all_params; }
  const std::vector<Parameters*>& parameters_list() const { return params; }
  const std::vector<LookupParameters*>& lookup_parameters_list() const { return lookup_params; }

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    int np = params.size();
    int nlp = lookup_params.size();
    ar & np;
    ar & nlp;
    for (unsigned i = 0; i < params.size(); ++i)
      ar & *params[i];
    for (unsigned i = 0; i < lookup_params.size(); ++i)
      ar & *lookup_params[i];
  }
  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    int np, nlp;
    ar & np;
    ar & nlp;
    assert(np == (int)params.size());
    assert(nlp == (int)lookup_params.size());
    for (unsigned i = 0; i < params.size(); ++i)
      ar & *params[i];
    for (unsigned i = 0; i < lookup_params.size(); ++i)
      ar & *lookup_params[i];
    all_params.clear();
    for (auto p : params) all_params.push_back(p);
    for (auto p : lookup_params) all_params.push_back(p);
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

  std::vector<ParametersBase*> all_params;
  std::vector<Parameters*> params;
  std::vector<LookupParameters*> lookup_params;
};

} // namespace cnn

#endif
