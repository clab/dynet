/**
 * \file model.h
 * \defgroup params params
 *
 */

#ifndef DYNET_PARAMS_H_
#define DYNET_PARAMS_H_

#include <memory>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <stdexcept>

#include "dynet/weight-decay.h"
#include "dynet/tensor.h"
// #include "dynet/devices.h"
#include "dynet/globals.h"

namespace dynet {

// to deal with sparse updates, there are two parameter classes:
// * Parameters represents a vector, matrix, (eventually higher order tensors)
//   of parameters. These are densely updated.
// * LookupParameters represents a table of vectors that are used to embed a
//   set of discrete objects. These are sparsely updated.

class DeviceManager;
class ParameterCollection;
struct ParameterInit;

/**
 * \ingroup params
 * @brief This is the base class for ParameterStorage and LookupParameterStorage, the objects handling the actual parameters.
 * @details You can access the storage from any Parameter (resp. LookupParameter) class, use it only to do low level manipulations.
 *
 */
struct ParameterStorageBase {
  friend class ParameterCollection;
  /**
   * @brief Scale the parameters
   *
   * @param a scale factor
   */
  virtual void scale_parameters(float a) = 0;
  /**
   * @brief Scale the gradient
   *
   * @param a scale factor
   */
  virtual void scale_gradient(float a) = 0;
  /**
   * @brief Set the parameters to 0
   */
  virtual void zero() = 0;
  /**
   * @brief Get the parameter squared l2 norm
   *
   * @param sqnorm Pointer to the float holding the result
   */
  virtual void squared_l2norm(float* sqnorm) const = 0;
  /**
   * @brief Get the squared l2 norm of the gradient w.r.t. these parameters
   *
   * @param sqnorm Pointer to the float holding the result
   */
  virtual void g_squared_l2norm(float* sqnorm) const = 0;
  /**
   * @brief Check whether corpus is updated
   *
   */
  virtual bool is_updated() const = 0;
  /**
   * @brief Check whether the gradient is zero or not (true if gradient is non-zero)
   *
   */
  virtual bool has_grad() const = 0;
  /**
   * @brief Get the size (number of scalar parameters)
   * @return Number of scalar parameters
   */
  virtual size_t size() const = 0;
  virtual ~ParameterStorageBase();
}; // struct ParameterStorageBase

// represents parameters (e.g., a weight matrix) that will be optimized
/**
 * \ingroup params
 * \brief Storage class for Parameters
 */
struct ParameterStorage : public ParameterStorageBase {
  friend class ParameterCollection;
  template <class MyDevice>
  void scale_parameters_dev(MyDevice & dev, float a);
  void scale_parameters(float a) override;
  template <class MyDevice>
  void scale_gradient_dev(MyDevice & dev, float a);
  void scale_gradient(float a) override;
  void zero() override;
  template <class MyDevice>
  void squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void squared_l2norm(float* sqnorm) const override;
  template <class MyDevice>
  void g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const;
  void g_squared_l2norm(float* sqnorm) const override;
  size_t size() const override;
  /**
   * @brief Copy from another ParameterStorage
   *
   * @param val ParameterStorage to copy from
   */
  void copy(const ParameterStorage & val);
  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, const Tensor& g);
  /**
   * @brief Add a tensor to the gradient
   * @details After this method gets called, g <- g + d
   *
   * @param g Tensor to add
   */
  void accumulate_grad(const Tensor& g);
  /**
   * @brief Clear the gradient (set it to 0)
   */
  void clear();

  bool is_updated() const override { return updated; }
  bool has_grad() const override { return nonzero_grad; }

  std::string name; /**< Name of this parameter*/

  /**
   * @brief Clip the values to the range [left, right]
   */
  void clip(float left, float right);
  void set_value(const std::vector<float>& val);
  

  Dim dim; /**< Dimensions of the parameter tensor*/
  Tensor values;/**< Values of the parameter */
  Tensor g;/**< Values of the gradient w.r.t. this parameter */
  bool updated; /**< Whether this is updated */
  bool nonzero_grad; /**< Whether the gradient is zero */
  ParameterCollection* owner; /**< Pointer to the collection that "owns" this parameter */
  Device *device;

protected:
  ParameterStorage() : updated(true), owner(nullptr) {}
  explicit ParameterStorage(const Dim& d, float scale,
                            const std::string & name, Device *device); // initialize with a scale
  explicit ParameterStorage(const Dim& d, const ParameterInit & init,
                            const std::string & name, Device *device); // initialize with custom initializer
}; // struct ParameterStorage

struct ParameterStorageCreator : public ParameterStorage {
  template <typename... Args>
  explicit ParameterStorageCreator(Args &&...args)
    : ParameterStorage(std::forward<Args>(args)...) {}

  template <typename... Args>
  static std::shared_ptr<ParameterStorage> create(Args &&...args) {
    return std::make_shared<ParameterStorageCreator>(std::forward<Args>(args)...);
  }
};

// represents a matrix/vector embedding of a discrete set
/**
 * \ingroup params
 * \brief Storage class for LookupParameters
 * 
 */
struct LookupParameterStorage : public ParameterStorageBase {
  friend class ParameterCollection;
  template <class MyDevice>
  void scale_parameters_dev(MyDevice & dev, float a);
  void scale_parameters(float a) override;
  template <class MyDevice>
  void scale_gradient_dev(MyDevice & dev, float a);
  void scale_gradient(float a) override;
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
  /**
   * @brief Initialize one particular lookup
   * 
   * @param index Index of the lookput to initialize
   * @param val Values
   */
  void initialize(unsigned index, const std::vector<float>& val);

  /**
   * @brief Copy from another LookupParameterStorage
   * 
   * @param val Other LookupParameterStorage to copy from
   */
  void copy(const LookupParameterStorage & val);

  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, const Tensor& g);
  /**
   * @brief Add a Tensor to the gradient of the whole lookup matrix
   * @details after this `grads<-grads + g`
   * 
   * @param g [description]
   */
  void accumulate_grad(const Tensor& g);

  template <class MyDevice>
  void accumulate_grad_dev(MyDevice & dev, unsigned index, const Tensor& g);
  /**
   * @brief Add a Tensor to the gradient of one of the lookups
   * @details after this `grads[index]<-grads[index] + g`
   * 
   * @param index [description]
   * @param g [description]
   */
  void accumulate_grad(unsigned index, const Tensor& g);
  template <class MyDevice>
  void accumulate_grads_dev(MyDevice & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
  /**
   * @brief Add tensors to muliple lookups
   * @details After this method gets called, `grads[ids_host[i]] <- grads[ids_host[i]] + g[i*dim.size():(i+1)*dim.size()]`
   *
   * @param n size of `ids_host`
   * @param ids_host Indices of the gradients to update
   * @param ids_dev [To be documented] (only for GPU)
   * @param g Values
   */
  void accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
  void clear();

  // Initialize each individual lookup from the overall tensors
  void initialize_lookups();

  bool is_updated() const override { return updated; }
  bool has_grad() const override { return nonzero_grad; }

  std::string name; /**< Name of this parameter*/
  // Tensors for all dimensions at once
  Dim all_dim; /**< Total dimension */
  Tensor all_values; /**< Values for all dimensions at once */
  Tensor all_grads; /**< Gradient values for all dimensions at once */
  // Tensors for each individual lookup
  Dim dim; /**< Dimension for one lookup */
  std::vector<Tensor> values; /**< List of values for each lookup */
  std::vector<Tensor> grads; /**< List of gradient values for each lookup */
  // gradients are sparse, so track which components are nonzero
  std::unordered_set<unsigned> non_zero_grads; /**< Gradients are sparse, so track which components are nonzero */
  bool updated; /**< Whether this lookup parameter should be updated */
  bool all_updated; /** Whether all of the gradients have been updated. */
  bool nonzero_grad; /**< Whether the gradient is zero */
  ParameterCollection* owner; /**< Pointer to the collection that "owns" this parameter */
  Device *device;
protected:
  LookupParameterStorage() : updated(true), all_updated(false), owner(nullptr) {}
  LookupParameterStorage(unsigned n, const Dim& d, const ParameterInit & init,
                         const std::string & name, Device *device);
}; // struct LookupParameterStorage

struct LookupParameterStorageCreator : public LookupParameterStorage {
  template <typename... Args>
  explicit LookupParameterStorageCreator(Args &&...args)
    : LookupParameterStorage(std::forward<Args>(args)...) {}

  template <typename... Args>
  static std::shared_ptr<LookupParameterStorage> create(Args &&...args) {
    return std::make_shared<LookupParameterStorageCreator>(std::forward<Args>(args)...);
  }
};

/**
 * \ingroup params
 * \brief Object representing a trainable parameter
 * \details This objects acts as a high level component linking the actual parameter values (ParameterStorage) and the ParameterCollection. As long as you don't want to do low level hacks at the ParameterStorage level, this is what you will use.
 *
 */
struct Parameter {
  /**
   * @brief Default constructor
   */
  Parameter();
  /**
   * @brief Constructor
   * @details This is called by the model, you shouldn't need to use it
   *
   * @param p Shared pointer to the parameter storage
   */
  Parameter(std::shared_ptr<ParameterStorage> p);
  /**
   * @brief Get underlying ParameterStorage object
   * @return ParameterStorage holding the parameter values
   */
  ParameterStorage& get_storage() const;

  /**
   * @brief Get the full name of the ParameterStorage object
   */
  std::string get_fullname() const;

  /**
   * \brief Zero the parameters
   */
  void zero();

  std::shared_ptr<ParameterStorage> p; /**< Pointer to the storage for this Parameter */

  /**
   * \brief Shape of the parameter
   *
   * \return Shape as a `Dim` object
   */
  Dim dim() const { return get_storage().dim; }

  /**
   * \brief Values of the parameter
   *
   * \return Values as a `Tensor` object
   */
  Tensor* values() { return &(get_storage().values); }

  /**
   * \brief gradients of the parameter
   *
   * \return gradients as a `Tensor` object
   */
  Tensor* gradients() { return &(get_storage().g); }

  /**
   * \brief Get the current weight decay for the parameters
   */
  float current_weight_decay() const;

  /**
   * @brief Set the parameter as updated
   *
   * @param b Update status
   */
  void set_updated(bool b);

  /**
   * @brief Scales the parameter (multiplies by `s`)
   *
   * @param s scale
   */
  void scale(float s){get_storage().scale_parameters(s);}


  /**
   * @brief Scales the gradient (multiplies by `s`)
   *
   * @param s scale
   */
  void scale_gradient(float s){get_storage().scale_gradient(s);}

  /**
   * @brief Check the update status
   * @return Update status
   */
  bool is_updated();

  /**
   * @brief Clip the values of the parameter to the range [left, right] (in place)
   */
  void clip_inplace(float left, float right);
  
  /**
  * @brief set the values of the parameter
  */
  void set_value(const std::vector<float>& val);

}; // struct Parameter


/**
 * \ingroup params
 * \brief Object representing a trainable lookup parameter
 *
 */
struct LookupParameter {
  LookupParameter();
  LookupParameter(std::shared_ptr<LookupParameterStorage> p);
  /**
   * @brief Get underlying LookupParameterStorage object
   * @return LookupParameterStorage holding the parameter values
   */
  LookupParameterStorage& get_storage() const;
  /**
   * @brief Initialize one particular column
   *
   * @param index Index of the column to be initialized
   * @param val [description]
   */
  void initialize(unsigned index, const std::vector<float>& val) const;

  /**
   * \brief Zero the parameters
   */
  void zero();

  std::shared_ptr<LookupParameterStorage> p; /**< Pointer to the storage for this Parameter */

  /**
   * @brief Get the full name of the ParameterStorage object
   */
  std::string get_fullname() const;

  /**
   * \brief Shape of the lookup parameter
   *
   * \return Shape as a `Dim` object
   */
  Dim dim() const { return get_storage().dim; }
  /**
   * \brief Values of the lookup parameter
   *
   * \return Values as a `Tensor` object
   */
  std::vector<Tensor>* values() { return &(get_storage().values); }

  /**
   * \brief Get the current weight decay for the parameters
   */
  float current_weight_decay() const;

  /**
   * @brief Scales the parameter (multiplies by `s`)
   *
   * @param s scale
   */
  void scale(float s){get_storage().scale_parameters(s);}

  /**
   * @brief Scales the gradient (multiplies by `s`)
   *
   * @param s scale
   */
  void scale_gradient(float s){get_storage().scale_gradient(s);}

  /**
  * @brief Set the parameter as updated
  *
  * @param b Update status
  */
  void set_updated(bool b);
  /**
   * @brief Check the update status
   * @return Update status
   */
  bool is_updated();
}; // struct LookupParameter

// This is an internal class to store parameters in the collection
struct ParameterCollectionStorage {

  ParameterCollectionStorage();

  ~ParameterCollectionStorage();

  void project_weights(float radius = 1.0f);

  template <class MyDevice>
  float gradient_l2_norm_dev(MyDevice & dev) const;
  float gradient_l2_norm() const;

  std::vector<std::shared_ptr<ParameterStorageBase>> all_params;
  std::vector<std::shared_ptr<ParameterStorage>> params;
  std::vector<std::shared_ptr<LookupParameterStorage>> lookup_params;

  mutable float* gradient_norm_scratch;
  L2WeightDecay weight_decay;

 private:
  DeviceManager* const device_manager;
};

// this is a collection of parameters
// if you need a matrix of parameters, or a lookup table - ask an instance of this class
// this knows how to serialize itself
// parameters know how to track their gradients, but any extra information (like velocity) will live here
/**
 * \ingroup params
 * \brief This is a collection of parameters
 * \details if you need a matrix of parameters, or a lookup table - ask an instance of this class.
 * This knows how to serialize itself.
 * Parameters know how to track their gradients, but any extra information (like velocity) will live here
 */
class ParameterCollection {
public:
  friend struct Parameter;
  friend struct LookupParameter;

  /**
   * \brief Constructor
   */
  ParameterCollection();
  ~ParameterCollection();
  /**
   * \brief Returns the l2 of your gradient
   * \details Use this to look for gradient vanishing/exploding
   * \return L2 norm of the gradient
   */
  float gradient_l2_norm() const;
  /**
   * \brief Sets all gradients to zero
   */
  void reset_gradient();
  // set scale to use custom initialization
  /**
   * \brief Add parameters to model and returns Parameter object
   * \details creates a ParameterStorage object holding a tensor of dimension `d` and returns a Parameter object (to be used as input in the computation graph). The coefficients are sampled according to the `scale` parameter
   *
   * \param d Shape of the parameter
   * \param scale If scale is non-zero, initializes according to \f$mathcal U([-\mathrm{scale},+\mathrm{scale}]\f$, otherwise uses Glorot initialization
   * \param name Name of the parameter
   * \param device Device placement for the parameter
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter add_parameters(const Dim& d, float scale = 0.0f,
                           const std::string & name = "", Device *device = dynet::default_device);
  /**
   * \brief Add parameters to model and returns Parameter object
   * \details creates a ParameterStorage object holding a tensor of dimension `d` and returns a Parameter object (to be used as input in the computation graph).
   *
   * \param d Shape of the parameter
   * \param device Device placement for the parameter
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter add_parameters(const Dim& d, Device *device);
  /**
   * \brief Add parameters to model and returns Parameter object
   * \details creates a ParameterStorage object holding a tensor of dimension `d` and returns a Parameter object (to be used as input in the computation graph).
   *
   * \param d Shape of the parameter
   * \param name Name of the parameter
   * \param device Device placement for the parameter
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter add_parameters(const Dim& d, const std::string & name, Device *device = dynet::default_device);
  /**
   * \brief Add parameters with custom initializer
   *
   * \param d Shape of the parameter
   * \param init Custom initializer
   * \param name Name of the parameter
   * \param device Device placement for the parameter
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter add_parameters(const Dim& d, const ParameterInit & init,
                           const std::string & name = "", Device *device = dynet::default_device);
  /**
   * \brief Get parameters base in current model
   *
   * \return list of points to ParameterStorageBase objects
   */
  std::vector<std::shared_ptr<ParameterStorageBase>> get_parameter_storages_base() const;
  /**
   * \brief Get parameter in current model
   * \details It is not recommended to use this
   * \return the pointer to the Parameter object
   */
  std::shared_ptr<ParameterStorage> get_parameter_storage(const std::string & pname);
  /**
   * \brief Get parameters in current model
   *
   * \return list of points to ParameterStorage objects
   */
  std::vector<std::shared_ptr<ParameterStorage>> get_parameter_storages() const;
  /**
   * \brief Add lookup parameter to model
   * \details Same as add_parameters. Initializes with Glorot
   *
   * \param n Number of lookup indices
   * \param d Dimension of each embedding
   * \param name Name of the parameter
   * \param device Device placement for the parameter
   *
   * \return LookupParameter object to be used in the computation graph
   */
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d,
                                        const std::string & name = "", Device *device = dynet::default_device);
  /**
   * \brief Add lookup parameter with custom initializer
   *
   * \param n Number of lookup indices
   * \param d Dimension of each embedding
   * \param init Custom initializer
   * \param name Name of the parameter
   * \param device Device placement for the parameter
   *
   * \return LookupParameter object to be used in the computation graph
   */
  LookupParameter add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init,
                                        const std::string & name = "", Device *device = dynet::default_device);
  /**
   * \brief Get lookup parameter in current model
   * \details It is not recommended to use this
   * \return the pointer to the LookupParameter object
   */
  std::shared_ptr<LookupParameterStorage> get_lookup_parameter_storage(const std::string & lookup_pname);
  /**
   * \brief Get lookup parameters in current model
   *
   * \return list of points to LookupParameterStorage objects
   */
  std::vector<std::shared_ptr<LookupParameterStorage>> get_lookup_parameter_storages() const;
  //
  /**
   * \brief project weights so their L2 norm = radius
   * \details NOTE (Paul) : I am not sure this is doing anything currently. The argument doesn't seem to be used anywhere... If you need this raise an issue on github
   *
   * \param radius Target norm
   */
  void project_weights(float radius = 1.0f);
  /**
   * \brief Set the weight decay coefficient
   *
   * \param lambda Weight decay coefficient
   */
  void set_weight_decay_lambda(float lambda);

  //const std::vector<std::shared_ptr<ParameterStorageBase>>& all_parameters_list() const { return all_params; }
  /**
   * \brief Returns list of shared pointers to ParameterSorages
   * \details You shouldn't need to use this
   * \return List of shared pointers to ParameterSorages
   */
  const std::vector<std::shared_ptr<ParameterStorage>>& parameters_list() const { return get_storage().params; }
  /**
   * \brief Returns list of pointers to LookupParameterSorages
   * \details You shouldn't need to use this
   * \return List of pointers to LookupParameterSorages
   */
  const std::vector<std::shared_ptr<LookupParameterStorage>>& lookup_parameters_list() const { return get_storage().lookup_params; }

  //
  //
  /**
   * \brief Returns the total number of tunable parameters (i. e. scalars) contained within this model.
   * \details That is to say, a 2x2 matrix counts as four parameters.
   * \return Number of parameters
   */
  size_t parameter_count() const;
  /**
   * \brief Returns total number of (scalar) parameters updated
   *
   * \return number of updated parameters
   */
  size_t updated_parameter_count() const;

  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \param status [description]
   */
  void set_updated_param(const Parameter *p, bool status);
  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \param status [description]
   */
  void set_updated_lookup_param(const LookupParameter *p, bool status);
  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \return [description]
   */
  bool is_updated_param(const Parameter *p);
  /**
   * \brief [brief description]
   * \details [long description]
   *
   * \param p [description]
   * \return [description]
   */
  bool is_updated_lookup_param(const LookupParameter *p);
  /**
   * \brief Add a sub-collection
   * \details This will allow you to add a ParameterCollection that is a
   *          (possibly named) subset of the original collection. This is
   *          useful if you want to save/load/update only part of the
   *          parameters in the model.
   * \return The subcollection
   */
  ParameterCollection add_subcollection(const std::string& name = "");

  /**
   * \brief Get size
   * \details Get the number of parameters in the ParameterCollection
   */
  size_t size() { return get_parameter_storages().size(); }

  /**
   * @brief get namespace of current ParameterCollection object(end with a slash)
   */
  std::string get_fullname() const { return name; }

  /**
   * \brief Get the weight decay object
   */
  L2WeightDecay& get_weight_decay() { return get_storage().weight_decay; }

  ParameterCollectionStorage& get_storage();
  const ParameterCollectionStorage& get_storage() const;

protected:
  void add_parameters_to_storage(std::shared_ptr<ParameterStorage> p);
  void add_lookup_parameters_to_storage(std::shared_ptr<LookupParameterStorage> p);

private:
  ParameterCollection(const std::string & name, ParameterCollection* parent);
  std::string name;
  std::unordered_map<std::string,int> name_cntr, collec_name_cntr;
  ParameterCollectionStorage * storage;
  ParameterCollection * parent;
}; // class ParameterCollection

void save_dynet_model(std::string filename, ParameterCollection* model);
void load_dynet_model(std::string filename, ParameterCollection* model);

class Model : public ParameterCollection {
public:
  Model();
};

} // namespace dynet


#endif
