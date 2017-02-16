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
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "model.h"
#include "tensor.h"
#include "dynet.h"
#include "training.h"
#include "expr.h"
#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "fast-lstm.h"
%}

//
// Macro to generate extra vector constructors that take a java Collection,
// needs to be declared + used before we include "std_vector.i"
//

%define VECTORCONSTRUCTOR(ctype, javatype, vectortype)
%typemap(javacode) std::vector<ctype> %{
  public vectortype(java.util.Collection<javatype> values) {
     this(values.size());
     int i = 0;
     for (java.util.Iterator<javatype> it = values.iterator(); it.hasNext(); i++) {
         javatype value = it.next();
         this.set(i, value);
     }
  }
%}
%enddef

VECTORCONSTRUCTOR(float, Float, FloatVector)
VECTORCONSTRUCTOR(double, Double, DoubleVector)
VECTORCONSTRUCTOR(int, Integer, IntVector)
VECTORCONSTRUCTOR(unsigned, Integer, UnsignedVector)
VECTORCONSTRUCTOR(dynet::expr::Expression, Expression, ExpressionVector)
VECTORCONSTRUCTOR(dynet::Parameter, Parameter, ParameterVector)
VECTORCONSTRUCTOR(std::vector<dynet::expr::Expression>, ExpressionVector, ExpressionVectorVector)
VECTORCONSTRUCTOR(std::vector<dynet::Parameter>, ParameterVector, ParameterVectorVector)


// Useful SWIG libraries
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"
%include "cpointer.i"

%pointer_functions(unsigned, uintp);
%pointer_functions(int, intp);
%pointer_functions(float, floatp);

struct dynet::expr::Expression;

// Declare explicit types for needed instantiations of generic types
namespace std {
  %template(IntVector)                    vector<int>;
  %template(UnsignedVector)               vector<unsigned>;
  %template(DoubleVector)                 vector<double>;
  %template(FloatVector)                  vector<float>;
  %template(LongVector)                   vector<long>;
  %template(StringVector)                 vector<std::string>;
  %template(ExpressionVector)             vector<dynet::expr::Expression>;
  %template(ParameterStorageVector)       vector<dynet::ParameterStorage*>;
  %template(LookupParameterStorageVector) vector<dynet::LookupParameterStorage*>;
  %template(ExpressionVectorVector)       vector<vector<dynet::expr::Expression>>;
  %template(ParameterVector)              vector<dynet::Parameter>;
  %template(ParameterVectorVector)        vector<vector<dynet::Parameter>>;
}

//
// The subset of classes/methods/functions we want to wrap
//

namespace dynet {

// Some declarations etc to keep swig happy
typedef float real;
typedef int RNNPointer;
struct VariableIndex;
/*{
  unsigned t;
  explicit VariableIndex(const unsigned t_): t(t_) {};
};*/
struct Tensor;
struct Node;
struct ParameterStorage;
struct LookupParameterStorage;

///////////////////////////////////
// declarations from dynet/dim.h //
///////////////////////////////////

%rename(get) Dim::operator[];

%typemap(javacode) Dim %{
  public Dim(long... values) {
    this();

    int i = 0;
    for (long l: values) {
      this.resize(i + 1);
      this.set(i, l);
      i++;
    }
  }

  @Override
  public boolean equals(Object obj) {
    // must be the same class
    if (obj instanceof $javaclassname) {
      $javaclassname other = ($javaclassname)obj;
      // must have the same shapes
      if (this.ndims() != other.ndims() ||
          this.batch_elems() != other.batch_elems()) return false;

      // must have the same values for every dim
      for (int i = 0; i < this.ndims(); i++) {
        if (this.get(i) != other.get(i)) return false;
      }

      return true;
    }
    return false;
  }

  @Override
  public int hashCode() {
    int hash = 17 * (int)this.ndims() + (int)this.batch_elems();
    for (int i = 0; i < this.ndims(); i++) {
      hash = hash * 31 + (int)this.get(i);
    }
    return hash;
  }
%}

struct Dim {
  Dim() : nd(0), bd(1) {}
  Dim(const std::vector<long> & x);
  Dim(const std::vector<long> & x, unsigned int b);

  unsigned int size();
  unsigned int batch_size();
  unsigned int sum_dims();

  Dim truncate();
  Dim single_batch();

  void resize(unsigned int i);
  unsigned int ndims();
  unsigned int rows();
  unsigned int cols();
  unsigned int batch_elems();
  void set(unsigned int i, unsigned int s);
  unsigned int operator[](unsigned int i);
  unsigned int size(unsigned int i);

  void delete_dim(unsigned int i);

  Dim transpose();
};

/////////////////////////////////////
// declarations from dynet/model.h //
/////////////////////////////////////

// Model wrapper class needs to implement Serializable. We serialize a Model by converting it
// to/from a String and using writeObject/readObject on the String.
%typemap(javainterfaces) dynet::Model "java.io.Serializable"

%typemap(javacode) dynet::Model %{
 private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
    out.defaultWriteObject();
    String s = this.serialize_to_string();
    out.writeObject(s);
 }

 private void readObject(java.io.ObjectInputStream in)
     throws java.io.IOException, java.lang.ClassNotFoundException {
    in.defaultReadObject();
    String s = (String) in.readObject();

    // Deserialization doesn't call the constructor, so the swigCPtr is 0. This means we need to
    // do the constructor work ourselves if we don't want a segfault.
    if (this.swigCPtr == 0) {
        this.swigCPtr = dynet_swigJNI.new_Model();
        this.swigCMemOwn = true;
    }

    this.load_from_string(s);
 }
%}

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

struct ParameterInit {
  ParameterInit() {}
  virtual ~ParameterInit() {}
  virtual void initialize_params(Tensor & values) const = 0;
};

struct ParameterInitNormal : public ParameterInit {
  ParameterInitNormal(float m = 0.0f, float v = 1.0f) : mean(m), var(v) {}
  virtual void initialize_params(Tensor& values) const override;
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

/* I AM NOT ACTUALLY IMPLEMENTED IN THE DYNET CODE
struct ParameterInitSaxe : public ParameterInit {
  ParameterInitSaxe() {}
  virtual void initialize_params(Tensor & values) const override;
private:
  float cnst;
};
*/

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


struct ParameterStorageBase {
  virtual void scale_parameters(float a) = 0;
  virtual void zero() = 0;
  virtual void squared_l2norm(float* sqnorm) const = 0;
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  virtual size_t size() const = 0;
  virtual ~ParameterStorageBase();
};

%nodefaultctor ParameterStorage;
struct ParameterStorage : public ParameterStorageBase {
  void scale_parameters(float a) override;
  void zero() override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;

  void copy(const ParameterStorage & val);
  void accumulate_grad(const Tensor& g);
  void clear();

  Dim dim;
  Tensor values;
  Tensor g;
};

%nodefaultctor LookupParameterStorage;
struct LookupParameterStorage : public ParameterStorageBase {
  void scale_parameters(float a) override;
  void zero() override;
  void squared_l2norm(float* sqnorm) const override;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  void initialize(unsigned index, const std::vector<float>& val);

  void copy(const LookupParameterStorage & val);
  void accumulate_grad(unsigned index, const Tensor& g);
  void accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
  void clear();

  // Initialize each individual lookup from the overall tensors
  void initialize_lookups();
};

// extra code for serialization and parameter lookup
%extend Model {
   std::string serialize_to_string() {
       std::ostringstream out;
       boost::archive::text_oarchive oa(out);
       oa << (*($self));
       return out.str();
   }

   void load_from_string(std::string serialized) {
       std::istringstream in;
       in.str(serialized);
       boost::archive::text_iarchive ia(in);
       ia >> (*($self));
   }

   // SWIG can't get the types right for `parameters_list`, so here are replacement methods
   // for which it can. (You might worry that these would cause infinite recursion, but
   // apparently they don't.
   std::vector<ParameterStorage*> parameters_list() const {
     return $self->parameters_list();
   }

   std::vector<LookupParameterStorage*> lookup_parameters_list() const {
     return $self->lookup_parameters_list();
   }

   std::vector<unsigned> updated_parameters_list() const {
     return $self->updated_parameters_list();
   }

   std::vector<unsigned> updated_lookup_parameters_list() const {
     return $self->updated_lookup_parameters_list();
   }
};

class Model {
 public:
  Model();
  ~Model();
  float gradient_l2_norm() const;
  void reset_gradient();

  Parameter add_parameters(const Dim& d, float scale = 0.0f);
  Parameter add_parameters(const Dim& d, const ParameterInit & init);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d);
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init);

  void project_weights(float radius = 1.0f);
  void set_weight_decay_lambda(float lambda);

  size_t parameter_count() const;
  size_t updated_parameter_count() const;

  void set_updated_param(const Parameter *p, bool status);
  void set_updated_lookup_param(const LookupParameter *p, bool status);
  bool is_updated_param(const Parameter *p);
  bool is_updated_lookup_param(const LookupParameter *p);
};

void save_dynet_model(std::string filename, Model* model);
void load_dynet_model(std::string filename, Model* model);

//////////////////////////////////////
// declarations from dynet/tensor.h //
//////////////////////////////////////

struct Tensor {
  Dim d;
  float* v;
  std::vector<Tensor> bs;
};

real as_scalar(const Tensor& t);
std::vector<real> as_vector(const Tensor& v);

struct TensorTools {
  static float AccessElement(const Tensor& v, const Dim& index);
};

/////////////////////////////////////
// declarations from dynet/nodes.h //
/////////////////////////////////////

struct Sum;
struct LogSumExp;
struct AffineTransform;
struct ConcatenateColumns;
struct Concatenate;
struct Average;

////////////////////////////////////
// declarations from dynet/expr.h //
////////////////////////////////////


struct ComputationGraph;

namespace expr {
struct Expression {
  ComputationGraph *pg;
  VariableIndex i;
  Expression(ComputationGraph *pg, VariableIndex i) : pg(pg), i(i) { };
  const Tensor& value();
  const Dim& dim() const { return pg->get_dimension(i); }
};

// This template gets used to instantiate operations on vector<Expression>
namespace detail {
template <typename F, typename T> Expression f(const T& xs);
}

/* INPUT OPERATIONS */

Expression input(ComputationGraph& g, real s);
Expression input(ComputationGraph& g, const real *ps);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<float>* pdata);
Expression input(ComputationGraph& g, const Dim& d, const std::vector<unsigned int>& ids, const std::vector<float>& data, float defdata = 0.f);
Expression parameter(ComputationGraph& g, Parameter p);
Expression const_parameter(ComputationGraph& g, Parameter p);
Expression lookup(ComputationGraph& g, LookupParameter p, unsigned index);
Expression lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
Expression const_lookup(ComputationGraph& g, LookupParameter p, unsigned index);
Expression const_lookup(ComputationGraph& g, LookupParameter p, const unsigned* pindex);
Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
//Expression lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);
//Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>& indices);
Expression const_lookup(ComputationGraph& g, LookupParameter p, const std::vector<unsigned>* pindices);

Expression zeroes(ComputationGraph& g, const Dim& d);
Expression random_normal(ComputationGraph& g, const Dim& d);
Expression random_bernoulli(ComputationGraph& g, const Dim& d, real p, real scale = 1.0f);
Expression random_uniform(ComputationGraph& g, const Dim& d, real left, real right);

/* ARITHMETIC OPERATIONS */

// Rename operators to valid Java function names
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

// TODO(joelgrus) rename these without the VE
%template(affine_transform_VE) detail::f<AffineTransform, std::vector<Expression>>;
%template(sum) detail::f<Sum, std::vector<Expression>>;
%template(average) detail::f<Average, std::vector<Expression>>;

Expression sqrt(const Expression& x);
Expression erf(const Expression& x);
Expression tanh(const Expression& x);
Expression exp(const Expression& x);
Expression square(const Expression& x);
Expression cube(const Expression& x);
Expression lgamma(const Expression& x);
Expression log(const Expression& x);
Expression logistic(const Expression& x);
Expression rectify(const Expression& x);
Expression softsign(const Expression& x);
Expression pow(const Expression& x, const Expression& y);

Expression min(const Expression& x, const Expression& y);

// We need two overloaded versions of `max`, but apparently %template
// gets unhappy when you use it to overload a function, so we have to define
// the `ExpressionVector` version of `max` explicitly.
%{
namespace dynet { namespace expr {
Expression max(const std::vector<Expression>& xs) {
  return detail::f<Max, std::vector<Expression>>(xs);
};
} }
%}

Expression max(const Expression& x, const Expression& y);
Expression max(const std::vector<Expression>& xs);
Expression dot_product(const Expression& x, const Expression& y);
Expression cmult(const Expression& x, const Expression& y);
Expression cdiv(const Expression& x, const Expression& y);
Expression colwise_add(const Expression& x, const Expression& bias);

/* PROBABILITY / LOSS OPERATIONS */

Expression softmax(const Expression& x);
Expression log_softmax(const Expression& x);
Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);

%template(logsumexp) detail::f<LogSumExp, std::vector<Expression>>;

// TODO(joelgrus): delete this once no one is using it
%template(logsumexp_VE) detail::f<LogSumExp, std::vector<Expression>>;

Expression pickneglogsoftmax(const Expression& x, unsigned v);
Expression pickneglogsoftmax(const Expression& x, const unsigned* pv);
Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned>& v);

Expression hinge(const Expression& x, unsigned index, float m = 1.0);
Expression hinge(const Expression& x, unsigned* pindex, float m = 1.0);
Expression hinge(const Expression& x, const std::vector<unsigned>& indices, float m = 1.0);

Expression sparsemax(const Expression& x);
Expression sparsemax_loss(const Expression& x, const std::vector<unsigned>& target_support);

Expression squared_norm(const Expression& x);
Expression squared_distance(const Expression& x, const Expression& y);
Expression l1_distance(const Expression& x, const Expression& y);
Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);
Expression binary_log_loss(const Expression& x, const Expression& y);
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m = 1.0);
Expression poisson_loss(const Expression& x, unsigned y);
Expression poisson_loss(const Expression& x, const unsigned* py);

/* FLOW / SHAPING OPERATIONS */

Expression nobackprop(const Expression& x);
Expression reshape(const Expression& x, const Dim& d);
Expression transpose(const Expression& x);
Expression select_rows(const Expression& x, const std::vector<unsigned> &rows);
Expression select_cols(const Expression& x, const std::vector<unsigned> &cols);
Expression sum_batches(const Expression& x);

Expression pick(const Expression& x, unsigned v, unsigned d = 0);
Expression pick(const Expression& x, const std::vector<unsigned>& v, unsigned d = 0);
Expression pick(const Expression& x, const unsigned* v, unsigned d = 0);
Expression pickrange(const Expression& x, unsigned v, unsigned u);

%template(concatenate_cols) detail::f<ConcatenateColumns, std::vector<Expression>>;
%template(concatenate) detail::f<Concatenate, std::vector<Expression>>;

// TODO(joelgrus): delete these once no one is using them
%template(concatenate_cols_VE) detail::f<ConcatenateColumns, std::vector<Expression>>;
%template(concatenate_VE) detail::f<Concatenate, std::vector<Expression>>;

/* NOISE OPERATIONS */

Expression noise(const Expression& x, real stddev);
Expression dropout(const Expression& x, real p);
Expression block_dropout(const Expression& x, real p);

/* CONVOLUTION OPERATIONS */

Expression conv1d_narrow(const Expression& x, const Expression& f);
Expression conv1d_wide(const Expression& x, const Expression& f);
Expression filter1d_narrow(const Expression& x, const Expression& f);
Expression kmax_pooling(const Expression& x, unsigned k);
Expression fold_rows(const Expression& x, unsigned nrows=2);
Expression sum_dim(const Expression& x, unsigned d);
Expression sum_cols(const Expression& x);
Expression sum_rows(const Expression& x);
Expression average_cols(const Expression& x);
Expression kmh_ngram(const Expression& x, unsigned n);

/* TENSOR OPERATIONS */

Expression contract3d_1d(const Expression& x, const Expression& y);
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z);
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const
 Expression& b);
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);

/* LINEAR ALGEBRA OPERATIONS */

Expression inverse(const Expression& x);
Expression logdet(const Expression& x);
Expression trace_of_product(const Expression& x, const Expression& y);

} // namespace expr

/////////////////////////////////////
// declarations from dynet/dynet.h //
/////////////////////////////////////

%typemap(javacode) ComputationGraph %{
  // DyNet only allows one ComputationGraph at a time. This means that if you construct them
  // manually you have to remember to delete each one before you construct a new one, or your
  // program will crash. `getNew` will handle that deletion for you.
  private static ComputationGraph singletonInstance = null;

  public static ComputationGraph getNew() {
    if (singletonInstance != null) {
      singletonInstance.delete();
    }
    singletonInstance = new ComputationGraph();
    return singletonInstance;
  }
%}

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

  Dim& get_dimension(VariableIndex index) const;

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

////////////////////////////////////////
// declarations from dynet/training.h //
////////////////////////////////////////

// Need to disable constructor as SWIG gets confused otherwise
%nodefaultctor Trainer;
struct Trainer {
  void update(real scale = 1.0);
  void update_epoch(real r = 1);

  float clip_gradients(real scale);
  void rescale_and_reset_weight_decay();

  real eta0;
  real eta;
  real eta_decay;
  real epoch;

  bool clipping_enabled;
  real clip_threshold;
  real clips;
  real updates;

  real clips_since_status;
  real updates_since_status;

  bool sparse_updates_enabled;
  bool aux_allocated;

  void status();

  Model* model;
};

struct SimpleSGDTrainer : public Trainer {
  explicit SimpleSGDTrainer(Model& m, real e0 = 0.1, real edecay = 0.0);
};

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(Model& m, real e0 = 0.01, real mom = 0.9, real edecay = 0.0);
};

struct AdagradTrainer : public Trainer {
  explicit AdagradTrainer(Model& m, real e0 = 0.1, real eps = 1e-20, real edecay = 0.0);
};

struct AdadeltaTrainer : public Trainer {
  explicit AdadeltaTrainer(Model& m, real eps = 1e-6, real rho = 0.95, real edecay = 0.0);
};

struct RmsPropTrainer : public Trainer {
   explicit RmsPropTrainer(Model& m, real e0 = 0.1, real eps = 1e-20, real rho = 0.95, real edecay = 0.0);
};

struct AdamTrainer : public Trainer {
  explicit AdamTrainer(Model& m, float e0 = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8, real edecay = 0.0);
};

///////////////////////////////////
// declarations from dynet/rnn.h //
///////////////////////////////////

%nodefaultctor RNNBuilder;
struct RNNBuilder {
  using namespace dynet::expr;

  RNNPointer state() const;
  void new_graph(ComputationGraph& cg);
  void start_new_sequence(const std::vector<Expression>& h_0 = {});
  Expression set_h(const RNNPointer& prev, const std::vector<Expression>& h_new = {});
  Expression set_s(const RNNPointer& prev, const std::vector<Expression>& s_new = {});
  Expression add_input(const Expression& x);
  Expression add_input(const RNNPointer& prev, const Expression& x);
  void rewind_one_step();
  RNNPointer get_head(const RNNPointer& p);
  void set_dropout(float d);
  void disable_dropout();

  virtual Expression back() const;
  virtual std::vector<Expression> final_h() const = 0;
  virtual std::vector<Expression> get_h(RNNPointer i) const = 0;

  virtual std::vector<Expression> final_s() const = 0;
  virtual std::vector<Expression> get_s(RNNPointer i) const = 0;

  virtual unsigned num_h0_components() const = 0;
  virtual void copy(const RNNBuilder& params) = 0;
  virtual void save_parameters_pretraining(const std::string& fname) const;
  virtual void load_parameters_pretraining(const std::string& fname);
};

struct SimpleRNNBuilder : public RNNBuilder {
  using namespace dynet::expr;
  SimpleRNNBuilder() = default;

  explicit SimpleRNNBuilder(unsigned layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
                            Model& model,
                            bool support_lags = false);

  Expression add_auxiliary_input(const Expression& x, const Expression& aux);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;
  void copy(const RNNBuilder& params) override;

  unsigned num_h0_components() const override;

  void save_parameters_pretraining(const std::string& fname) const override;
  void load_parameters_pretraining(const std::string& fname) override;
};

////////////////////////////////////
// declarations from dynet/lstm.h //
////////////////////////////////////

struct LSTMBuilder : public RNNBuilder {
  using namespace dynet::expr;

  LSTMBuilder() = default;
  explicit LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model& model);
  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder& params) override;

  void save_parameters_pretraining(const std::string& fname) const override;
  void load_parameters_pretraining(const std::string& fname) override;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
};

struct VanillaLSTMBuilder : public RNNBuilder {
  VanillaLSTMBuilder() = default;
  explicit VanillaLSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model& model);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder & params) override;

  void save_parameters_pretraining(const std::string& fname) const override;
  void load_parameters_pretraining(const std::string& fname) override;

  // first index is layer, then ...
  std::vector<std::vector<Parameter>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
  unsigned hid;
};

///////////////////////////////////
// declarations from dynet/gru.h //
///////////////////////////////////

struct GRUBuilder : public RNNBuilder {
  GRUBuilder() = default;
  explicit GRUBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model& model);
  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;
  unsigned num_h0_components() const override;
  void copy(const RNNBuilder & params) override;
};


/////////////////////////////////////////
// declarations from dynet/fast-lstm.h //
/////////////////////////////////////////

struct FastLSTMBuilder : public RNNBuilder {
  FastLSTMBuilder() = default;
  explicit FastLSTMBuilder(unsigned layers,
                           unsigned input_dim,
                           unsigned hidden_dim,
                           Model& model);

  Expression back() const override;
  std::vector<Expression> final_h() const override;
  std::vector<Expression> final_s() const override;
  unsigned num_h0_components() const override;

  std::vector<Expression> get_h(RNNPointer i) const override;
  std::vector<Expression> get_s(RNNPointer i) const override;

  void copy(const RNNBuilder & params) override;

  std::vector<std::vector<Parameter>> params;
  std::vector<std::vector<Expression>> param_vars;

  std::vector<std::vector<Expression>> h, c;

  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
};

////////////////////////////////////
// declarations from dynet/init.h //
////////////////////////////////////

struct DynetParams {
  unsigned random_seed = 0; /**< The seed for random number generation */
  std::string mem_descriptor = "512"; /**< Total memory to be allocated for Dynet */
  float weight_decay = 0; /**< Weight decay rate for L2 regularization */
  bool shared_parameters = false; /**< TO DOCUMENT */

#ifdef SWIG_USE_CUDA
  bool ngpus_requested = false; /**< GPUs requested by number */
  bool ids_requested = false; /**< GPUs requested by ids */
  int requested_gpus = -1; /**< Number of requested GPUs */
  std::vector<int> gpu_mask; /**< List of required GPUs by ids */
#endif
};


void initialize(DynetParams params);
void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();


//////////////////////////////////////////////////
// serialization logic (from python/pybridge.h) //
//////////////////////////////////////////////////

// Add Java method to ModelSaver for serializing java objects.
%typemap(javaimports) ModelSaver %{
  import java.io.ByteArrayOutputStream;
  import java.io.ObjectOutputStream;
  import java.io.IOException;
%}
 
%typemap(javacode) ModelSaver %{
  public void add_object(Object o) {
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ObjectOutputStream objOut = new ObjectOutputStream(out);
      objOut.writeObject(o);
      objOut.close();

      byte[] bytes = out.toByteArray();

      add_size(bytes.length);
      add_byte_array(bytes);
    } catch (IOException e) {
      // This shouldn't ever happen.
      throw new RuntimeException(e);
    }
  }
%}

// Add Java method to ModelLoader for loading java objects. 
%typemap(javaimports) ModelLoader %{
  import java.io.ByteArrayInputStream;
  import java.io.ObjectInputStream;
  import java.io.IOException;
%}

%typemap(javacode) ModelLoader %{
  public <T> T load_object(Class<T> clazz) {
    long size = load_size();
    byte[] bytes = new byte[(int) size];
    load_byte_array(bytes);

    Object obj = null;
    try {
      ByteArrayInputStream in = new ByteArrayInputStream(bytes);
      ObjectInputStream objIn = new ObjectInputStream(in);
      obj = objIn.readObject();
      objIn.close();
    } catch (IOException e) {
      // This shouldn't ever happen.
      throw new RuntimeException(e);
    } catch (ClassNotFoundException e) {
      // This also shouldn't happen (because the class is an argument).
      throw new RuntimeException(e);
    }

    return clazz.cast(obj);
  }
%}


%{
  
namespace dynet {

struct ModelSaver {
    ModelSaver(std::string filename) : ofs(filename), oa(ofs) {}

    void add_model(Model& model) { oa << model; }
    void add_parameter(Parameter &p) { oa << p; }
    void add_lookup_parameter(LookupParameter &p) { oa << p; }
    void add_rnn_builder(RNNBuilder &p) { oa << p; }
    void add_lstm_builder(LSTMBuilder &p) { oa << p; }
    void add_vanilla_lstm_builder(VanillaLSTMBuilder &p) { oa << p; }
    void add_srnn_builder(SimpleRNNBuilder &p) { oa << p; }
    void add_gru_builder(GRUBuilder &p) { oa << p; }
    void add_fast_lstm_builder(FastLSTMBuilder &p) { oa << p; }
    void add_size(size_t len) { oa << len; }
    void add_byte_array(char *str, size_t len) {
      oa << boost::serialization::make_array(str, len);
    }

    // primitive types
    void add_int(int x) { oa << x; }
    void add_long(jlong x) { oa << x; }
    void add_float(float x) { oa << x; }
    void add_double(double x) { oa << x; }
    void add_boolean(jboolean x) { oa << x; }

    void done() { ofs.close(); }

    private:
        std::ofstream ofs;
        boost::archive::text_oarchive oa;

};

struct ModelLoader {
    ModelLoader(std::string filename) : ifs(filename), ia(ifs) {}

    Model* load_model() {
      Model* model = new Model(); ia >> *model; return model;
    }

    Parameter* load_parameter() {
      Parameter* p = new Parameter(); ia >> *p; return p;
    }

    LookupParameter* load_lookup_parameter() {
      LookupParameter* p = new LookupParameter(); ia >> *p; return p;
    }

    LSTMBuilder* load_lstm_builder() {
      LSTMBuilder* p = new LSTMBuilder(); ia >> *p; return p;
    }

    VanillaLSTMBuilder* load_vanilla_lstm_builder() {
      VanillaLSTMBuilder* p = new VanillaLSTMBuilder() ; ia >> *p; return p;
    }

    SimpleRNNBuilder* load_srnn_builder() {
      SimpleRNNBuilder* p = new SimpleRNNBuilder(); ia >> *p; return p;
    }

    GRUBuilder* load_gru_builder() {
      GRUBuilder* p = new GRUBuilder(); ia >> *p; return p;
    }

    FastLSTMBuilder* load_fast_lstm_builder() {
      FastLSTMBuilder* p = new FastLSTMBuilder(); ia >> *p; return p;
    }

    size_t load_size() {
      size_t len;
      ia >> len;
      return len;
    }

    void load_byte_array(char *str, size_t len) {
      ia >> boost::serialization::make_array(str, len);
    }

    int load_int() { int x; ia >> x; return x; }
    jlong load_long() { long x; ia >> x; return x; }
    float load_float() { float x; ia >> x; return x; }
    double load_double() { double x; ia >> x; return x; }
    jboolean load_boolean() { bool x; ia >> x; return x; }

    void done() { ifs.close(); }

    private:
        std::ifstream ifs;
        boost::archive::text_iarchive ia;

};

}
%}

// Convert methods whose arguments are (char *str, size_t len)
// to byte[] in Java. Note that the *argument names must match*,
// not just the types.
%apply(char *STRING, size_t LENGTH) { (char *str, size_t len) }

%nodefaultctor ModelSaver;
struct ModelSaver {
    ModelSaver(std::string filename);
    void add_model(Model& model);
    void add_parameter(Parameter &p);
    void add_lookup_parameter(LookupParameter &p);
    void add_lstm_builder(LSTMBuilder &p);
    void add_vanilla_lstm_builder(VanillaLSTMBuilder &p);
    void add_srnn_builder(SimpleRNNBuilder &p);
    void add_gru_builder(GRUBuilder &p);
    void add_fast_lstm_builder(FastLSTMBuilder &p);
    void add_size(size_t len) { oa << len; }
    void add_byte_array(char *str, size_t len);

    void add_int(int x);
    void add_long(jlong x);
    void add_float(float x);
    void add_double(double x);
    void add_boolean(jboolean x);

    void done();
};

%newobject ModelLoader::load_model();
%newobject ModelLoader::load_parameter();
%newobject ModelLoader::load_lookup_parameter();
%newobject ModelLoader::load_rnn_builder();
%newobject ModelLoader::load_lstm_builder();
%newobject ModelLoader::load_vanilla_lstm_builder();
%newobject ModelLoader::load_srnn_builder();
%newobject ModelLoader::load_gru_builder();
%newobject ModelLoader::load_fast_lstm_builder();


%nodefaultctor ModelLoader;
struct ModelLoader {
    ModelLoader(std::string filename);
    Model* load_model();
    Parameter* load_parameter();
    LookupParameter* load_lookup_parameter();
    LSTMBuilder* load_lstm_builder();
    VanillaLSTMBuilder* load_vanilla_lstm_builder();
    SimpleRNNBuilder* load_srnn_builder();
    GRUBuilder* load_gru_builder();
    FastLSTMBuilder* load_fast_lstm_builder();
    size_t load_size();
    void load_byte_array(char *str, size_t len);

    int load_int();
    jlong load_long();
    float load_float();
    double load_double();
    jboolean load_boolean();

    void done();
};

}




