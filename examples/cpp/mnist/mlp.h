#ifndef MLP_H
#define MLP_H

/**
 * \file rnnlm-batch.h
 * \defgroup ffbuilders ffbuilders
 * \brief Feed forward nets builders
 *
 * An example implementation of a simple multilayer perceptron
 *
 */

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/expr.h"
#include "dynet/io-macros.h"

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dynet;
using namespace dynet::expr;
/**
 * \ingroup ffbuilders
 * Common activation functions used in multilayer perceptrons
 */
enum Activation {
  SIGMOID, /**< `SIGMOID` : Sigmoid function \f$x\longrightarrow \frac {1} {1+e^{-x}}\f$ */
  TANH, /**< `TANH` : Tanh function \f$x\longrightarrow \frac {1-e^{-2x}} {1+e^{-2x}}\f$ */
  RELU, /**< `RELU` : Rectified linear unit \f$x\longrightarrow \max(0,x)\f$ */
  LINEAR, /**< `LINEAR` : Identity function \f$x\longrightarrow x\f$ */
  SOFTMAX /**< `SOFTMAX` : Softmax function \f$\textbf{x}=(x_i)_{i=1,\dots,n}\longrightarrow \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j} })_{i=1,\dots,n}\f$ */
};

/**
 * \ingroup ffbuilders
 * \struct Layer
 * \brief Simple layer structure
 * \details Contains all parameters defining a layer
 *
 */
struct Layer {
public:
  unsigned input_dim; /**< Input dimension */
  unsigned output_dim; /**< Output dimension */
  Activation activation = LINEAR; /**< Activation function */
  float dropout_rate = 0; /**< Dropout rate */
  /**
   * \brief Build a feed forward layer
   *
   * \param input_dim Input dimension
   * \param output_dim Output dimension
   * \param activation Activation function
   * \param dropout_rate Dropout rate
   */
  Layer(unsigned input_dim, unsigned output_dim, Activation activation, float dropout_rate) :
    input_dim(input_dim),
    output_dim(output_dim),
    activation(activation),
    dropout_rate(dropout_rate) {};
  Layer() {};
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int) {
    ar & input_dim & output_dim & activation & dropout_rate;
  }
};
DYNET_SERIALIZE_IMPL(Layer);

/**
 * \ingroup ffbuilders
 * \struct MLP
 * \brief Simple multilayer perceptron
 *
 */
struct MLP {
protected:
  // Hyper-parameters
  unsigned LAYERS = 0;

  // Layers
  vector<Layer> layers;
  // Parameters
  vector<vector<Parameter>> params;

  bool dropout_active = true;

public:
  /**
   * \brief Default constructor
   * \details Dont forget to add layers!
   */
  MLP(Model & model) {
    LAYERS = 0;
  }
  /**
   * \brief Returns a Multilayer perceptron
   * \details Creates a feedforward multilayer perceptron based on a list of layer descriptions
   *
   * \param model Model to contain parameters
   * \param layers Layers description
   */
  MLP(Model& model,
      vector<Layer> layers) {
    // Verify layers compatibility
    for (unsigned l = 0; l < layers.size() - 1; ++l) {
      if (layers[l].output_dim != layers[l + 1].input_dim)
        throw invalid_argument("Layer dimensions don't match");
    }

    // Register parameters in model
    for (Layer layer : layers) {
      append(model, layer);
    }
  }

  /**
   * \brief Append a layer at the end of the network
   * \details [long description]
   *
   * \param model [description]
   * \param layer [description]
   */
  void append(Model& model, Layer layer) {
    // Check compatibility
    if (LAYERS > 0)
      if (layers[LAYERS - 1].output_dim != layer.input_dim)
        throw invalid_argument("Layer dimensions don't match");

    // Add to layers
    layers.push_back(layer);
    LAYERS++;
    // Register parameters
    Parameter W = model.add_parameters({layer.output_dim, layer.input_dim});
    Parameter b = model.add_parameters({layer.output_dim});
    params.push_back({W, b});
  }

  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
  Expression run(Expression x,
                 ComputationGraph& cg) {
    // Expression for the current hidden state
    Expression h_cur = x;
    for (unsigned l = 0; l < LAYERS; ++l) {
      // Initialize parameters in computation graph
      Expression W = parameter(cg, params[l][0]);
      Expression b = parameter(cg, params[l][1]);
      // Aplly affine transform
      Expression a = affine_transform({b, W, h_cur});
      // Apply activation function
      Expression h = activate(a, layers[l].activation);
      // Take care of dropout
      Expression h_dropped;
      if (layers[l].dropout_rate > 0) {
        if (dropout_active) {
          // During training, drop random units
          Expression mask = random_bernoulli(cg, {layers[l].output_dim}, 1 - layers[l].dropout_rate);
          h_dropped = cmult(h, mask);
        } else {
          // At test time, multiply by the retention rate to scale
          h_dropped = h * (1 - layers[l].dropout_rate);
        }
      } else {
        // If there's no dropout, don't do anything
        h_dropped = h;
      }
      // Set current hidden state
      h_cur = h_dropped;
    }

    return h_cur;
  }

  /**
   * \brief Return the negative log likelihood for the (batched) pair (x,y)
   * \details For a batched input \f$\{x_i\}_{i=1,\dots,N}\f$, \f$\{y_i\}_{i=1,\dots,N}\f$, this computes \f$\sum_{i=1}^N \log(P(y_i\vert x_i))\f$ where \f$P(\textbf{y}\vert x_i)\f$ is modelled with $\mathrm{softmax}(MLP(x_i))$
   *
   * \param x Input batch
   * \param labels Output labels
   * \param cg Computation graph
   * \return Expression for the negative log likelihood on the batch
   */
  Expression get_nll(Expression x,
                     vector<unsigned> labels,
                     ComputationGraph& cg) {
    // compute output
    Expression y = run(x, cg);
    // Do softmax
    Expression losses = pickneglogsoftmax(y, labels);
    // Sum across batches
    return sum_batches(losses);
  }

  /**
   * \brief Predict the most probable label
   * \details Returns the argmax of the softmax of the networks output
   *
   * \param x Input
   * \param cg Computation graph
   *
   * \return Label index
   */
  int predict(Expression x,
              ComputationGraph& cg) {
    // run MLP to get class distribution
    Expression y = run(x, cg);
    // Get values
    vector<float> probs = as_vector(cg.forward(y));
    // Get argmax
    unsigned argmax = 0;
    for (unsigned i = 1; i < probs.size(); ++i) {
      if (probs[i] > probs[argmax])
        argmax = i;
    }

    return argmax;
  }

  /**
   * \brief Enable dropout
   * \details This is supposed to be used during training or during testing if you want to sample outputs using montecarlo
   */
  void enable_dropout() {
    dropout_active = true;
  }

  /**
   * \brief Disable dropout
   * \details Do this during testing if you want a deterministic network
   */
  void disable_dropout() {
    dropout_active = false;
  }

  /**
   * \brief Check wether dropout is enabled or not
   *
   * \return Dropout state
   */
  bool is_dropout_enabled() {
    return dropout_active;
  }

private:
  inline Expression activate(Expression h, Activation f) {
    switch (f) {
    case LINEAR:
      return h;
      break;
    case RELU:
      return rectify(h);
      break;
    case SIGMOID:
      return logistic(h);
      break;
    case TANH:
      return tanh(h);
      break;
    case SOFTMAX:
      return softmax(h);
      break;
    default:
      throw invalid_argument("Unknown activation function");
      break;
    }
  }

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int) {
    ar & LAYERS;
    ar & layers & params;
    ar & dropout_active;
  }


};

#endif