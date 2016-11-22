#ifndef PROTO_SERIALIZATION_UTIL_H_
#define PROTO_SERIALIZATION_UTIL_H_

#include "dynet/dynet.h"
#include "tensor.pb.h"

using namespace dynet;

class SerializationUtil {
 public:
  static void AsL2WeightDecayProto(const L2WeightDecay &weight_decay, L2WeightDecayProto *proto) {
    proto->set_weight_decay(weight_decay.weight_decay);
    proto->set_lambda(weight_decay.lambda);
  }

  static void FromL2WeightDecayProto(const L2WeightDecayProto &proto, L2WeightDecay *weight_decay) {
    weight_decay->weight_decay = proto.weight_decay();
    weight_decay->lambda = proto.lambda();
  }

  static void AsDimensionsProto(const Dim &dimensions, DimensionsProto *proto) {
    for (unsigned i = 0; i < dimensions.nd; ++i) {
      proto->add_dimension(dimensions.d[i]);
    }
  }

  static void FromDimensionsProto(const DimensionsProto &proto, Dim *dimensions) {
    dimensions->nd = proto.dimension_size();
    for (int i = 0; i < proto.dimension_size(); ++i) {
      dimensions->d[i] = proto.dimension(i);
    }
  }

  static void AsTensorProto(const Tensor &tensor, TensorProto *proto) {
    AsDimensionsProto(tensor.d, proto->mutable_dimensions());
    for (unsigned i = 0; i < tensor.d.size(); ++i) {
      proto->add_value(tensor.v[i]);
    }
  }

  // Assumes that the mutable tensor is preallocated to the right size.
  static void FromTensorProto(const TensorProto &proto, Tensor *tensor) {
    FromDimensionsProto(proto.dimensions(), &tensor->d);
    assert(tensor->d.size() == static_cast<unsigned>(proto.value_size()));
    for (int i = 0; i < proto.value_size(); ++i) {
      tensor->v[i] = proto.value(i);
    }
  }

  static void AsParameterProto(const ParameterStorage &parameter, ParameterProto *proto) {
    AsDimensionsProto(parameter.dim, proto->mutable_dimensions());
    AsTensorProto(parameter.values, proto->add_value());
  }

  static void AsParameterProto(const LookupParameterStorage &parameter, ParameterProto *proto) {
    AsDimensionsProto(parameter.dim, proto->mutable_dimensions());
    for (const Tensor &value : parameter.values) {
      AsTensorProto(value, proto->add_value());
    }
  }

  static void FromParameterProto(const ParameterProto &proto, ParameterStorage *parameter) {
    FromDimensionsProto(proto.dimensions(), &parameter->dim);
    assert(proto.value_size() == 1);
    FromTensorProto(proto.value(0), &parameter->values);
  }

  static void FromParameterProto(const ParameterProto &proto, LookupParameterStorage *parameter) {
    FromDimensionsProto(proto.dimensions(), &parameter->dim);
    for (int i = 0; i < proto.value_size(); ++i) {
      FromTensorProto(proto.value(i), &parameter->values[i]);
    }
  }

  static void AsModelProto(const Model &model, ModelProto *proto) {
    for (ParameterStorage *parameter : model.parameters_list()) {
      AsParameterProto(*parameter, proto->add_parameter());
    }
    for (LookupParameterStorage *parameter : model.lookup_parameters_list()) {
      AsParameterProto(*parameter, proto->add_lookup_parameter());
    }
    AsL2WeightDecayProto(model.weight_decay, proto->mutable_weight_decay());
  }

  static void FromModelProto(const ModelProto &proto, Model *model) {
    assert(static_cast<int>(model->parameters_list().size()) == proto.parameter_size());
    for (int i = 0; i < proto.parameter_size(); ++i) {
      Dim dimensions;
      FromDimensionsProto(proto.parameter(i).dimensions(), &dimensions);
      assert(dimensions == model->parameters_list()[i]->dim);
      FromParameterProto(proto.parameter(i), model->parameters_list()[i]);
    }
    assert(static_cast<int>(model->lookup_parameters_list().size()) == proto.lookup_parameter_size());
    for (int i = 0; i < proto.lookup_parameter_size(); ++i) {
      Dim dimensions;
      FromDimensionsProto(proto.lookup_parameter(i).dimensions(), &dimensions);
      assert(dimensions == model->lookup_parameters_list()[i]->dim);
      FromParameterProto(proto.lookup_parameter(i), model->lookup_parameters_list()[i]);
    }
    FromL2WeightDecayProto(proto.weight_decay(), &model->weight_decay);
  }

  static void Save(const std::string &model_file, const Model &model) {
    ModelProto model_proto;
    SerializationUtil::AsModelProto(model, &model_proto);
    std::fstream out_stream(model_file, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!model_proto.SerializeToOstream(&out_stream)) {
      throw std::runtime_error("Failed to serialize model.");
    }
    out_stream.close();
    std::cerr << "Saved model to " << model_file << std::endl;
  }

  static void Load(const std::string &model_file, Model *model) {
    ModelProto model_proto;
    std::fstream in_stream(model_file, std::ios::in | std::ios::binary);
    if (!model_proto.ParseFromIstream(&in_stream)) {
      throw std::runtime_error("Failed to load model.");
    }
    in_stream.close();
    SerializationUtil::FromModelProto(model_proto, model);
    std::cerr << "Loaded model from " << model_file << std::endl;
  }

};

#endif
