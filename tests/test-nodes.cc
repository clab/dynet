#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct NodeTest {
  NodeTest() {
    // initialize if necessary
    if(default_device == nullptr) {
      for (auto x : {"NodeTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }

    ones3_vals = {1.f,1.f,1.f};
    first_one_vals = {1.f,0.f,0.f};
    ones2_vals = {1.f,1.f};
    batch_vals = {1.f,2.f,3.f,4.f,5.f,6.f};
    // Create parameters
    std::vector<float> param1_vals = {1.1f,-2.2f,3.3f};
    std::vector<float> param2_vals = {2.2f,3.4f,-1.2f};
    std::vector<float> param3_vals = {1.1f,2.2f,3.3f};
    std::vector<float> param4_vals = {1.1f,2.2f,3.3f,-1.2f,2.1f,3.4f};
    std::vector<float> param_scalar1_vals = {2.2f};
    std::vector<float> param_scalar2_vals = {1.1f};
    std::vector<float> param_kernel1_vals = {1.1f,2.2f,-1.0f,1.2f,-3.4f,-0.2f};
    std::vector<float> param_filter1_vals = {1.1f,2.2f,-1.0f,1.2f,-3.4f,-0.2f,
                                             11.1f,12.2f,13.3f,11.2f,12.2f,13.2f};
    std::vector<float> param_square1_vals = {1.1f,2.2f,3.4f,1.2f,2.5f,3.2f,5.3f,2.3f,3.3f};
    std::vector<float> param_cube1_vals = {.011f,.022f,.033f,.012f,.022f,.032f,.013f,.023f,.033f,
                                           .111f,-.122f,-.033f,-.112f,-.022f,-.132f,-.113f,-.123f,-.133f,
                                           .211f,.222f,.233f,.212f,.222f,.232f,.213f,.223f,.233f};
    param1 = mod.add_parameters({3});
    TensorTools::SetElements(param1.get()->values,param1_vals);
    param2 = mod.add_parameters({3});
    TensorTools::SetElements(param2.get()->values,param2_vals);
    param3 = mod.add_parameters({3});
    TensorTools::SetElements(param3.get()->values,param3_vals);
    param4 = mod.add_parameters({6});
    TensorTools::SetElements(param4.get()->values,param4_vals);
    param_scalar1 = mod.add_parameters({1});
    TensorTools::SetElements(param_scalar1.get()->values,param_scalar1_vals);
    param_scalar2 = mod.add_parameters({1});
    TensorTools::SetElements(param_scalar2.get()->values,param_scalar2_vals);
    param_kernel1 = mod.add_parameters({3,2});
    TensorTools::SetElements(param_kernel1.get()->values,param_kernel1_vals);
    param_filter1 = mod.add_parameters({3,2,2});
    TensorTools::SetElements(param_filter1.get()->values,param_filter1_vals);
    param_square1 = mod.add_parameters({3,3});
    TensorTools::SetElements(param_square1.get()->values,param_square1_vals);
    param_cube1 = mod.add_parameters({3,3,3});
    TensorTools::SetElements(param_cube1.get()->values,param_cube1_vals);
    lookup1 = mod.add_lookup_parameters(3, {3});
    TensorTools::SetElements(lookup1.get()->all_values,param_square1_vals);
  }
  ~NodeTest() {
    for (auto x : av) free(x);
  }

  template <class T>
  std::string print_vec(const std::vector<T> vec) {
    ostringstream oss;
    if(vec.size()) oss << vec[0];
    for(size_t i = 1; i < vec.size(); i++)
      oss << ' ' << vec[i];
    return oss.str();
  }

  std::vector<float> ones3_vals, ones2_vals, first_one_vals, batch_vals;
  std::vector<char*> av;
  dynet::Model mod;
  dynet::Parameter param1, param2, param3, param4, param_scalar1, param_scalar2, param_kernel1, param_filter1, param_square1, param_cube1;
  dynet::LookupParameter lookup1;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(node_test, NodeTest);


// Expression operator-(const Expression& x);
BOOST_AUTO_TEST_CASE( negate_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = -x1;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( add_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1+x2;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( sum_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = sum({x2,x1,x2});
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( sum_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression x3 = input(cg, Dim({3},2), batch_vals);
  Expression y = sum({x3,x1,cmult(x2,x3)});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = sum_batches(ones3 * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logsumexp(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( logsumexp_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param_scalar2);
  Expression z = logsumexp({x1, x2});
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( addscalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1+2.0;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalaradd_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0+x1;
  Expression z =input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator-(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( subtract_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1+x2;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator-(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalarsubtract_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0-x1;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator-(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( subtractscalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1-2.0;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1*transpose(x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * y * transpose(ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = x1*transpose(x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = sum_batches(ones3 * y * transpose(ones3));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = affine_transform({x1, x2, scalar});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * sqrt(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = affine_transform({x1, x2, scalar});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = sum_batches(ones3 * sqrt(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch_col_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = input(cg, Dim({1,3},2), batch_vals);
  Expression y = affine_transform({transpose(x1), scalar, x2});
  Expression ones3 = input(cg, {3,1}, ones3_vals);
  Expression z = sum_batches(sqrt(y) * ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = input(cg, Dim({1,3},2), batch_vals);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = affine_transform({x1, scalar, transpose(x2) });
  Expression ones3 = input(cg, {3,1}, ones3_vals);
  Expression z = sum_batches(sqrt(y) * ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch3_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_square1);
  Expression inp = input(cg, Dim({3},2), batch_vals);
  Expression y = affine_transform({x1, x2, inp });
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = sum_batches(ones3 * sqrt(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, float y);
BOOST_AUTO_TEST_CASE( multiplyscalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1*2.0;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// inline Expression operator*(float y, const Expression& x) { return x * y; }
BOOST_AUTO_TEST_CASE( scalarmultiply_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0*x1;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }
BOOST_AUTO_TEST_CASE( dividescalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1/2.0;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cdiv(x1, x2);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = cdiv(x1, x2) + cdiv(x2, x1);
  Expression z = sum_batches(input(cg, {1,3}, ones3_vals) * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression colwise_add(const Expression& x, const Expression& bias);
BOOST_AUTO_TEST_CASE( colwise_add_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = colwise_add(x1 * transpose(x2), x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * y * transpose(ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate_cols(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_cols_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = concatenate_cols({x1, x2, x1});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * y * transpose(ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = transpose(parameter(cg, param1));
  Expression x2 = transpose(parameter(cg, param2));
  Expression y = concatenate({x1, x2, x1});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * y * transpose(ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = transpose(parameter(cg, param1));
  Expression x2 = transpose(parameter(cg, param2));
  Expression x3 = input(cg, Dim({1,3},2), batch_vals);
  Expression y = concatenate({x1, x2, cmult(x2, x3)});
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = sum_batches(ones3 * y * transpose(ones3));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
BOOST_AUTO_TEST_CASE( contract3d_1d_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression square1 = parameter(cg, param_square1);
  Expression cube1 = parameter(cg, param_cube1);
  Expression y = contract3d_1d(cube1, x1, square1);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * y * transpose(ones3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b);
BOOST_AUTO_TEST_CASE( contract3d_1d_1d_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression x3 = parameter(cg, param3);
  Expression cube1 = parameter(cg, param_cube1);
  Expression y = contract3d_1d_1d(cube1, x1, x2, x3);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  Expression z = ones3 * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sqrt(const Expression& x);
BOOST_AUTO_TEST_CASE( sqrt_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = sqrt(x3);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression erf(const Expression& x);
BOOST_AUTO_TEST_CASE( erf_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = erf(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression tanh(const Expression& x);
BOOST_AUTO_TEST_CASE( tanh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = tanh(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression exp(const Expression& x);
BOOST_AUTO_TEST_CASE( exp_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = exp(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression square(const Expression& x);
BOOST_AUTO_TEST_CASE( square_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = square(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cube(const Expression& x);
BOOST_AUTO_TEST_CASE( cube_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = cube(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression lgamma(const Expression& x);
BOOST_AUTO_TEST_CASE( lgamma_gradient ) {
  dynet::ComputationGraph cg;
  Expression x2 = parameter(cg, param2);
  Expression y = lgamma(x2);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log(const Expression& x);
BOOST_AUTO_TEST_CASE( log_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = log(x3);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logistic(const Expression& x);
BOOST_AUTO_TEST_CASE( logistic_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = logistic(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression rectify(const Expression& x);
BOOST_AUTO_TEST_CASE( rectify_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = rectify(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression hinge(const Expression& x, unsigned index, float m = 1.0);
BOOST_AUTO_TEST_CASE( hinge_gradient ) {
  unsigned index = 0;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = hinge(x1, index, 0.5);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression hinge(const Expression& x, unsigned index, float m = 1.0);
BOOST_AUTO_TEST_CASE( hinge_batch_gradient ) {
  std::vector<unsigned> idx = {1,2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression z = sum_batches(hinge(x1+x2, idx, 2.f));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);
BOOST_AUTO_TEST_CASE( hingeptr_gradient ) {
  unsigned index = 0;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = hinge(x1, &index, 0.5);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( log_softmax_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log_softmax(x1);
  Expression z = input(cg, {1,3}, first_one_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( log_softmax_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = log_softmax(x1+x2);
  Expression z = sum_batches(input(cg, {1,3}, first_one_vals) * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);
BOOST_AUTO_TEST_CASE( restricted_log_softmax_gradient ) {
  vector<unsigned> restriction = {0,1};
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = exp( log_softmax(x3, restriction) );
  Expression z = input(cg, {1,3}, first_one_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( softmax_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log(softmax(x1));
  Expression z = input(cg, {1,3}, first_one_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( softmax_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = log(softmax(x1+x2));
  Expression z = sum_batches(input(cg, {1,3}, first_one_vals) * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sparsemax(const Expression& x);
BOOST_AUTO_TEST_CASE( sparsemax_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = sparsemax(x1);
  Expression z = input(cg, {1,3}, first_one_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sparsemax_loss(const Expression& x);
BOOST_AUTO_TEST_CASE( sparsemax_loss_gradient ) {
  std::vector<unsigned> idxs(2); idxs[0] = 1; idxs[1] = 2;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = sparsemax_loss(x1, idxs);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softsign(const Expression& x);
BOOST_AUTO_TEST_CASE( softsign_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = softsign(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pow(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( pow_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression x_scalar1 = parameter(cg, param_scalar1);
  Expression y = pow(x3, x_scalar1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression min(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( min_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = min(x1, x2);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression max(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( max_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = max(x1, x2);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// TODO: Noise is random, so it cannot be tested simply?
// // Expression noise(const Expression& x, real stddev);
// BOOST_AUTO_TEST_CASE( noise_gradient ) {
//   dynet::ComputationGraph cg;
//   Expression x1 = parameter(cg, param1);
//   Expression y = noise(x1, 0.5);
//   input(cg, {1,3}, ones3_vals) * y;
//   BOOST_CHECK(check_grad(mod, cg, 0));
// }

// TODO: Dropout scales the gradients at training time, so they don't match.
// // Expression dropout(const Expression& x, real p);
// BOOST_AUTO_TEST_CASE( dropout_gradient ) {
//   dynet::ComputationGraph cg;
//   Expression x1 = parameter(cg, param1);
//   Expression y = dropout(x1, 0.5);
//   input(cg, {1,3}, ones3_vals) * y;
//   BOOST_CHECK(check_grad(mod, cg, 0));
// }

// TODO: Dropout scales the gradients at training time, so they don't match.
// Expression block_dropout(const Expression& x, real p);

// Expression reshape(const Expression& x, const Dim& d);
BOOST_AUTO_TEST_CASE( reshape_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = reshape(x1, {1,3});
  Expression z = y * input(cg, {3}, ones3_vals);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression reshape(const Expression& x, const Dim& d);
BOOST_AUTO_TEST_CASE( reshape_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y1 = x1*transpose(x2);
  Expression y2 = reshape(y1, Dim({3,3}, 2));
  Expression z = sum_batches(input(cg, {1,3}, ones3_vals) * y2 * input(cg, {3,1}, ones3_vals));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression transpose(const Expression& x);
BOOST_AUTO_TEST_CASE( transpose_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = softsign(x1);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// inverse is too numerically unstable to test appropriately
// // Expression inverse(const Expression& x);
// BOOST_AUTO_TEST_CASE( inverse_gradient ) {
//   dynet::ComputationGraph cg;
//   Expression x = parameter(cg, param_square1);
//   Expression y = inverse(x);
//   Expression z = input(cg, {1,3}, ones3_vals) * y * input(cg, {3,1}, ones3_vals);
//   BOOST_CHECK(check_grad(mod, z, 0));
// }

// Expression trace_of_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( trace_of_product_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = trace_of_product(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cmult(x1, x2);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression y = cmult(x1, x2) + cmult(x2, x1);
  Expression z = sum_batches(input(cg, {1,3}, ones3_vals) * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression dot_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( dot_product_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = dot_product(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression dot_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( dot_product_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression z = sum_batches(dot_product(x1, x2) + dot_product(x2, x1) * 2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_distance_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = squared_distance(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);
BOOST_AUTO_TEST_CASE( huber_distance_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = huber_distance(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression l1_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( l1_distance_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = l1_distance(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression binary_log_loss(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( binary_log_loss_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = logistic( parameter(cg, param1) );
  Expression x2 = input(cg, {3}, ones3_vals);
  Expression z = binary_log_loss(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m=1.0);
BOOST_AUTO_TEST_CASE( pairwise_rank_loss_gradient ) {
  dynet::ComputationGraph cg;
  Expression x_scalar1 = parameter(cg, param_scalar1);
  Expression x_scalar2 = parameter(cg, param_scalar2);
  Expression z = pairwise_rank_loss(x_scalar1, x_scalar2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression poisson_loss(const Expression& x, unsigned y);
BOOST_AUTO_TEST_CASE( possion_loss_gradient ) {
  dynet::ComputationGraph cg;
  Expression scalar = parameter(cg, param_scalar1);
  Expression z = poisson_loss(scalar, 3);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression conv1d_narrow(const Expression& x, const Expression& f);
BOOST_AUTO_TEST_CASE( conv1d_narrow_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression xkernel = parameter(cg, param_kernel1);
  Expression y = conv1d_narrow(xsquare, xkernel);
  Expression z = input(cg, {1,3}, ones3_vals) * y * input(cg, {2,1}, ones2_vals);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression conv1d_wide(const Expression& x, const Expression& f);
BOOST_AUTO_TEST_CASE( conv1d_wide_gradient ) {
  dynet::ComputationGraph cg;
  Expression xkernel = parameter(cg, param_kernel1);
  Expression y = conv1d_wide(xkernel, xkernel);
  Expression z = input(cg, {1,3}, ones3_vals) * y * input(cg, {3,1}, ones3_vals);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression filter1d_narrow(const Expression& x, const Expression& f);
BOOST_AUTO_TEST_CASE( filter1d_narrow_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression xfilter = parameter(cg, param_filter1);
  Expression y = filter1d_narrow(xsquare, xfilter);
  Expression z = input(cg, {1,2}, ones3_vals) * y * input(cg, {2,1}, ones2_vals);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression kmax_pooling(const Expression& x, unsigned k);
BOOST_AUTO_TEST_CASE( kmax_pooling_keq1_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(kmax_pooling(xsquare, 1));
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression kmax_pooling(const Expression& x, unsigned k);
BOOST_AUTO_TEST_CASE( kmax_pooling_keq2_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(kmax_pooling(xsquare, 2));
  Expression z = input(cg, {1,3}, ones3_vals) * y * input(cg, {2,1}, ones2_vals);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression fold_rows(const Expression& x, unsigned nrows=2);
BOOST_AUTO_TEST_CASE( fold_rows_gradient ) {
  dynet::ComputationGraph cg;
  Expression x4 = parameter(cg, param4);
  Expression y = fold_rows(x4, 2);
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression average_cols(const Expression& x);
BOOST_AUTO_TEST_CASE( average_cols_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(average_cols(xsquare));
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum_cols(const Expression& x);
BOOST_AUTO_TEST_CASE( sum_cols_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(sum_cols(xsquare));
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}


// TODO: These are all unimplemented
// Expression kmh_ngram(const Expression& x, unsigned n);

// Expression pick(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pick_gradient ) {
  unsigned idx = 1;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = pick(x1, idx);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick(const Expression& x, unsigned* pv);
BOOST_AUTO_TEST_CASE( pickptr_gradient ) {
  unsigned idx = 1;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = pick(x1, &idx);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pick_batch_gradient ) {
  std::vector<unsigned> idx = {1,2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression z = sum_batches(pick(x1+x2, idx));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pickrange(const Expression& x, unsigned v, unsigned u);
BOOST_AUTO_TEST_CASE( pickrange_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = pickrange(x1, 0, 2);
  Expression z = input(cg, {1,2}, ones2_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pickneglogsoftmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pickneglogsoftmax_gradient ) {
  unsigned idx = 1;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = pickneglogsoftmax(x1, idx);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pickneglogsoftmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pickneglogsoftmax_batch_gradient ) {
  std::vector<unsigned> idx = {1,2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3},2), batch_vals);
  Expression z = sum_batches(pickneglogsoftmax(x1+x2, idx));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sparse_input(vector<unsigned int>& ids, vector<float>& src, float def);
BOOST_AUTO_TEST_CASE( sparse_input_test ) {
  dynet::ComputationGraph cg;
  std::vector<unsigned int> ids = {0, 4};
  Expression z = input(cg, Dim({3},2), ids, ones2_vals, 0.5);
  std::vector<float> exp = {1.0f, 0.5f, 0.5f, 0.5f, 1.0f, 0.5f};
  std::vector<float> act = as_vector(cg.forward(z));
  assert(exp.size() == act.size());
  for(size_t i = 0; i < exp.size(); ++i)
    BOOST_CHECK_CLOSE(exp[i], act[i], 0.001);
}

// Expression lookup();
BOOST_AUTO_TEST_CASE( lookup_test ) {
  dynet::ComputationGraph cg;
  Expression x1 = lookup(cg, lookup1, (unsigned)0);
  Expression x2 = lookup(cg, lookup1, (unsigned)2);
  Expression y = x1+x2;
  Expression z = input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression lookup_seq()
BOOST_AUTO_TEST_CASE( lookup_seq_test ) {
  dynet::ComputationGraph cg;
  vector<unsigned> ids({0, 0, 2});
  Expression x = lookup_seq(cg, lookup1, ids);
  Expression z = input(cg, {1,3}, ones3_vals) * x * input(cg, {3}, ones3_vals);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression lookup_seq()
BOOST_AUTO_TEST_CASE( lookup_seq_batch_test ) {
  dynet::ComputationGraph cg;
  vector<vector<unsigned>> ids(2);
  ids[0] = {0, 2, 0};
  ids[0] = {1, 0, 2};
  Expression x1 = lookup_seq(cg, lookup1, ids);
  Expression x2 = parameter(cg, param1);
  Expression z = sum_batches( transpose(x2) * x1 * x2 );
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_SUITE_END()
