#include <chrono>
#include <fstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "dynet/dynet.h"
#include "proto/serialization-util.h"

using namespace dynet;

void assert_equal(const Tensor &t1, const Tensor &t2) {
  assert(t1.d == t2.d);
  for (unsigned i = 0; i < t1.d.size(); ++i) {
    assert(t1.v[i] == t2.v[i]);
  }
}

void assert_equal(const ParameterStorage &t1, const ParameterStorage &t2) {
  assert(t1.dim == t2.dim);
  assert_equal(t1.values, t2.values);
}

void assert_equal(const LookupParameterStorage &t1, const LookupParameterStorage &t2) {
  assert(t1.dim == t2.dim);
  assert(t1.values.size() == t2.values.size());
  for (unsigned i = 0; i < t1.values.size(); ++i) {
    assert_equal(t1.values[i], t2.values[i]);
  }
}

void assert_equal(const Model &m1, const Model &m2) {
  assert(m1.parameters_list().size() == m2.parameters_list().size());
  assert(m1.lookup_parameters_list().size() == m2.lookup_parameters_list().size());
  for (unsigned i = 0; i < m1.parameters_list().size(); ++i) {
    assert_equal(*m1.parameters_list()[i], *m2.parameters_list()[i]);
  }
  for (unsigned i = 0; i < m1.lookup_parameters_list().size(); ++i) {
    assert_equal(*m1.lookup_parameters_list()[i], *m2.lookup_parameters_list()[i]);
  }
}

void test_dimensions_serialization(const Dim &in_dim) {
  DimensionsProto dim_proto;
  SerializationUtil::AsDimensionsProto(in_dim, &dim_proto);

  Dim out_dim;
  SerializationUtil::FromDimensionsProto(dim_proto, &out_dim);
  assert(out_dim == in_dim);
}

void test_tensor_serialization(const Tensor &in_tensor) {
  TensorProto tensor_proto;
  SerializationUtil::AsTensorProto(in_tensor, &tensor_proto);

  Dim out_dim;
  SerializationUtil::FromDimensionsProto(tensor_proto.dimensions(), &out_dim);
  float *values = static_cast<float *>(_mm_malloc(out_dim.size() * sizeof(float), 32));
  Tensor out_tensor(out_dim, values, nullptr, DeviceMempool::NONE);
  SerializationUtil::FromTensorProto(tensor_proto, &out_tensor);
  assert_equal(in_tensor, out_tensor);
  _mm_free(values);
}

void test_model_serialization(const Model &in_model, const Dim &params_dim, const Dim &lookup_params_dim, int num_lookups) {
  ModelProto model_proto;
  SerializationUtil::AsModelProto(in_model, &model_proto);

  Model out_model;
  out_model.add_parameters(params_dim);
  out_model.add_lookup_parameters(num_lookups, lookup_params_dim);
  SerializationUtil::FromModelProto(model_proto, &out_model);
  assert_equal(in_model, out_model);
}

void do_testing() {
  Dim test_dim({500, 100});
  test_dimensions_serialization(test_dim);
  std::cerr << "Tested (de)serializing dimensions." << std::endl;

  float *values = static_cast<float *>(_mm_malloc(test_dim.size() * sizeof(float), 32));
  assert(values != nullptr);
  values[0] = 0;
  for (unsigned i = 0; i < test_dim.size(); ++i) {
    values[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  Tensor test_tensor(test_dim, values, nullptr, DeviceMempool::NONE);
  test_tensor_serialization(test_tensor);
  std::cerr << "Tested (de)serializing tensor." << std::endl;
  _mm_free(values);

  Dim params_dim({256, 128});
  Dim lookup_params_dim({128, 128});
  int num_lookups = 100;

  Model test_model;
  test_model.add_parameters(params_dim);
  test_model.add_lookup_parameters(num_lookups, lookup_params_dim);
  test_model_serialization(test_model, params_dim, lookup_params_dim, num_lookups);
  std::cerr << "Tested (de)serializing model." << std::endl;
}

void do_benchmarking() {
  Dim params_dim({1024, 512});
  Dim lookup_params_dim({128});
  int lookup_params_size = 2000;

  Model model;
  model.add_parameters(params_dim);
  model.add_lookup_parameters(lookup_params_size, lookup_params_dim);

  std::string boost_file = "model.ser";
  std::string protobuf_file = "model.pb";

  std::chrono::high_resolution_clock::time_point tic, toc;

  tic = std::chrono::high_resolution_clock::now();
  ModelProto output_model_proto;
  SerializationUtil::AsModelProto(model, &output_model_proto);
  std::fstream protobuf_out(protobuf_file, std::ios::out | std::ios::trunc | std::ios::binary);
  assert(output_model_proto.SerializeToOstream(&protobuf_out));
  protobuf_out.close();
  toc = std::chrono::high_resolution_clock::now();
  std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()
      << " ms. to save model with protobuffer." << std::endl;

  tic = std::chrono::high_resolution_clock::now();
  ModelProto input_model_proto;
  std::fstream protobuf_in(protobuf_file, std::ios::in | std::ios::binary);
  assert(input_model_proto.ParseFromIstream(&protobuf_in));
  protobuf_in.close();
  Model protobuf_model;
  protobuf_model.add_parameters(params_dim);
  protobuf_model.add_lookup_parameters(lookup_params_size, lookup_params_dim);
  SerializationUtil::FromModelProto(input_model_proto, &protobuf_model);
  toc = std::chrono::high_resolution_clock::now();
  std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()
      << " ms. to load model with protobuffer." << std::endl;

  assert_equal(model, protobuf_model);

  tic = std::chrono::high_resolution_clock::now();
  std::ofstream boost_out(boost_file);
  boost::archive::text_oarchive boost_oa(boost_out);
  boost_oa << model;
  boost_out.close();
  toc = std::chrono::high_resolution_clock::now();
  std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()
      << " ms. to save model with boost." << std::endl;

  tic = std::chrono::high_resolution_clock::now();
  Model boost_model;
  boost_model.add_parameters(params_dim);
  boost_model.add_lookup_parameters(lookup_params_size, lookup_params_dim);
  std::ifstream boost_in(boost_file);
  boost::archive::text_iarchive boost_ia(boost_in);
  boost_ia & boost_model;
  boost_in.close();
  toc = std::chrono::high_resolution_clock::now();
  std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()
      << " ms. to load model with boost." << std::endl;
  assert_equal(model, boost_model);
}

int main(int argc, char **argv) {
  initialize(argc, argv);
  do_testing();
  do_benchmarking();
}
