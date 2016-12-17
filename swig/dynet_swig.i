%module dynet_swig

// This module provides java bindings for the dynet C++ code

// Automatically load the library code
%pragma(java) jniclasscode=%{
    static {
        System.loadLibrary("dynet_swig");
    }
%}

// Required header files for compiling wrapped code
%{
#include <vector>
#include "model.h"
#include "tensor.h"
#include "dynet.h"
#include "training.h"
#include "expr.h"
#include "rnn.h"
#include "lstm.h"
%}

// Extra C++ code added
%{
namespace dynet {

// Convenience function for testing
static void myInitialize()  {
  char** argv = {NULL};
  int argc = 0;
  initialize(argc, argv);
};
}
%}

// Useful SWIG libraries
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"


struct dynet::expr::Expression;

// Declare explicit types for needed instantiations of generic types
namespace std {
  %template(IntVector)        vector<int>;
  %template(DoubleVector)     vector<double>;
  %template(FloatVector)      vector<float>;
  %template(LongVector)       vector<long>;
  %template(StringVector)     vector<std::string>;
  %template(ExpressionVector) vector<dynet::expr::Expression>;
}

//
// The subset of classes/methods/functions we want to wrap
//

namespace dynet {

// Some declarations etc to keep swig happy
typedef float real;

struct RNNPointer;
struct VariableIndex;
/*{
  unsigned t;
  explicit VariableIndex(const unsigned t_): t(t_) {};
};*/
struct Tensor;
struct Node;
struct ParameterStorage;
struct LookupParameterStorage;

struct Dim {
  Dim() : nd(0), bd(1) {}
  Dim(const std::vector<long> & x);
  Dim(const std::vector<long> & x, unsigned int b);
};

class Model;
struct Parameter {
  Parameter();
  Parameter(Model* mp, unsigned long index);
  void zero();
  Model* mp;
  unsigned long index;

  Dim dim();
  Tensor* values();

  void set_updated(bool b);
  bool is_updated();

};

struct LookupParameter {
  LookupParameter();
  LookupParameter(Model* mp, unsigned long index);
  LookupParameterStorage* get() const;
  void initialize(unsigned index, const std::vector<float>& val) const;
  void zero();
  Model* mp;
  unsigned long index;
  Dim dim() { return get()->dim; }
  std::vector<Tensor>* values() { return &(get()->values); }
  void set_updated(bool b);
  bool is_updated();
};

/*
struct LookupParameterStorage : public ParameterStorageBase {
  void scale_parameters(float a) override;
  void zero() override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  void initialize(unsigned index, const std::vector<float>& val);
  void accumulate_grad(unsigned index, const Tensor& g);
  void clear();
  void initialize_lookups();
  Dim all_dim;
  Tensor all_values;
  Tensor all_grads;
  Dim dim;
  std::vector<Tensor> values;
  std::vector<Tensor> grads;
  std::unordered_set<unsigned> non_zero_grads;
};
*/

class Model {
 public:
  Model();
  ~Model();
  float gradient_l2_norm() const;
  void reset_gradient();

  Parameter add_parameters(const Dim& d, float scale = 0.0f);
  // Parameter add_parameters(const Dim& d, const ParameterInit & init);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d);
  // LookupParameter add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init);

};

struct ComputationGraph;

struct Tensor {
  Dim d;
  float* v;
  std::vector<Tensor> bs;
};

real as_scalar(const Tensor& t);
std::vector<real> as_vector(const Tensor& v);

namespace expr {
struct Expression {
  ComputationGraph *pg;
  VariableIndex i;
  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i) { }
  //const Tensor& value() const { return pg->get_value(i); }
};

// %template(ExpressionVector)     ::std::vector<Expression>;

Expression input(ComputationGraph& g, real s);
Expression input(ComputationGraph& g, const real *ps);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>& data);
//Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);
Expression parameter(ComputationGraph& g, Parameter p);
Expression const_parameter(ComputationGraph& g, Parameter p);
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index);
//Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index);
//Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
//Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
//Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);
Expression zeroes(ComputationGraph& g, const Dim& d);
Expression random_normal(ComputationGraph& g, const Dim& d);

// Rename operators to valid java function names
%rename(exprPlus) operator+;
%rename(exprTimes) operator*;
%rename(exprMinus) operator-;
%rename(exprDivide) operator/;
Expression operator-(const Expression& x);
Expression operator+(const Expression& x, const Expression& y);
Expression operator+(const Expression& x, real y);
Expression operator+(real x, const Expression& y);
Expression operator-(const Expression& x, const Expression& y);
Expression operator-(real x, const Expression& y);
Expression operator-(const Expression& x, real y);
Expression operator*(const Expression& x, const Expression& y);
Expression operator*(const Expression& x, float y);
Expression operator*(float y, const Expression& x); // { return x * y; }
Expression operator/(const Expression& x, float y); // { return x * (1.f / y); }

Expression tanh(const Expression& x);
Expression squared_distance(const Expression& x, const Expression& y);

Expression noise(const Expression& x, real stddev);
Expression dropout(const Expression& x, real p);
Expression block_dropout(const Expression& x, real p);

template <typename T>
Expression affine_transform(const T& xs);
%template(affine_transform_VE) affine_transform<std::vector<Expression>>;

/*
template <typename T>
inline Expression affine_transform(const T& xs) { return detail::f<AffineTransform>(xs); }
inline Expression affine_transform(const std::initializer_list<Expression>& xs) { return detail::f<AffineTransform>(xs); }
void AffineTransform::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
*/


} // namespace expr


struct ComputationGraph {
  ComputationGraph();
  ~ComputationGraph();

  VariableIndex add_input(real s);
  // VariableIndex add_input(const real* ps);
  VariableIndex add_input(const Dim& d, const std::vector<float>& data);
  //VariableIndex add_input(const Dim& d, const std::vector<float>* pdata);
  VariableIndex add_input(const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);

  VariableIndex add_parameters(Parameter p);
  VariableIndex add_const_parameters(Parameter p);
  VariableIndex add_lookup(LookupParameter p, const unsigned* pindex);
  VariableIndex add_lookup(LookupParameter p, unsigned index);
  VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  // VariableIndex add_lookup(LookupParameter p, const std::vector<unsigned>& indices);
  VariableIndex add_const_lookup(LookupParameter p, const unsigned* pindex);
  VariableIndex add_const_lookup(LookupParameter p, unsigned index);
  VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>* pindices);
  // VariableIndex add_const_lookup(LookupParameter p, const std::vector<unsigned>& indices);

  void clear();
  void checkpoint();
  void revert();

  const Tensor& forward(const expr::Expression& last);
  //const Tensor& forward(VariableIndex i);
  const Tensor& incremental_forward(const expr::Expression& last);
  //const Tensor& incremental_forward(VariableIndex i);
  //const Tensor& get_value(VariableIndex i);
  const Tensor& get_value(const expr::Expression& e);
  void invalidate();
  void backward(const expr::Expression& last);
  //void backward(VariableIndex i);

  void print_graphviz() const;

  std::vector<Node*> nodes;
  std::vector<VariableIndex> parameter_nodes;
};


// Need to disable constructor as SWIG gets confused otherwise
%nodefaultctor Trainer;
struct Trainer {
  void update(real scale = 1.0);
  void update_epoch(real r = 1);
  void rescale_and_reset_weight_decay();
  real eta0;
  real eta;
  real eta_decay;
  real epoch;
  real clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;
  bool aux_allocated;
  Model* model;
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model* m, real e0 = 0.1, real edecay = 0.0) : Trainer(m, e0, edecay) {}
};


%nodefaultctor RNNBuilder;
struct RNNBuilder {
  RNNPointer state() const;
  void new_graph(ComputationGraph& cg);
  void start_new_sequence(const std::vector<Expression>& h_0 = {});
  Expression set_h(const RNNPointer& prev, const std::vector<Expression>& h_new = {});
  Expression set_s(const RNNPointer& prev, const std::vector<Expression>& s_new = {});
  Expression add_input(const Expression& x);
  Expression add_input(const RNNPointer& prev, const Expression& x);
};

struct LSTMBuilder : public RNNBuilder {
  //LSTMBuilder() = default;
  explicit LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);
};


void initialize(int& argc, char**& argv, bool shared_parameters = false);

static void myInitialize();
void cleanup();

}






