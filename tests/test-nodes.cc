#define BOOST_TEST_MODULE TEST_NODES

#include <dynet/functors.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <boost/test/unit_test.hpp>
#include "test.h"
#include <stdexcept>

using namespace dynet;
using namespace std;


struct NodeTest {
  NodeTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"NodeTest", "--dynet-mem", "100"}) {
        av.push_back(strdup(x));
      }
      ADD_EXTRA_ARGUMENTS(av)
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }

    ones3_vals = {1.f, 1.f, 1.f};
    first_one_vals = {1.f, 0.f, 0.f};
    ones2_vals = {1.f, 1.f};
    batch_vals = {1.1f, 2.6f, 3.3f, 4.0f, 5.1f, 6.6f};
    // Create parameters
    std::vector<float> param1_vals = {1.1f, -2.2f, 3.3f};
    std::vector<float> param2_vals = {2.2f, 3.4f, -1.2f};
    std::vector<float> param3_vals = {1.1f, 2.2f, 3.3f};
    std::vector<float> param4_vals = {1.1f, 2.2f, 3.3f, -1.2f, 2.1f, 3.4f};
    std::vector<float> param5_vals = {-0.2f, 0.0f, 0.1f};
    std::vector<float> param_scalar1_vals = {2.2f};
    std::vector<float> param_scalar2_vals = {1.1f};
    std::vector<float> param_kernel1_vals = {1.1f, 2.2f, -1.0f, 1.2f, -3.4f, -0.2f};
    std::vector<float> param_filter1_vals = {1.1f, 2.2f, -1.0f, 1.2f, -3.4f, -0.2f,
                                             11.1f, 12.2f, 13.3f, 11.2f, 12.2f, 13.2f
                                            };
    std::vector<float> param_square1_vals = {1.1f, 2.2f, 3.4f, 1.2f, 2.5f, 3.2f, 5.3f, 2.3f, 3.3f};
    std::vector<float> param_cube1_vals = {.051f, .062f, .073f, .052f, .062f, .072f, .053f, .063f, .073f,
                                           .111f, -.122f, -.033f, -.112f, -.022f, -.132f, -.113f, -.123f, -.133f,
                                           .211f, .222f, .233f, .212f, .222f, .232f, .213f, .223f, .233f
                                          };
    std::vector<float> param_cube2_vals = {
	.011f, 1.011f, .022f, 1.022f, .033f, 1.033f, .012f, 1.012f, .022f, 1.022f, .032f, 1.032f, .013f, 1.013f, .023f, 1.023f, .033f, 1.033f, // 18
	.111f, 1.111f, -.122f, -1.122f, -.033f, -1.033f, -.112f, -1.112f, -.022f, -1.022f, -.132f, -1.132f, -.113f, -1.113f, -.123f, -1.123f, -.133f, -1.133f, // 18
	.211f, 1.211f, .222f, 1.222f, .233f, 1.233f, .212f, 1.212f, .222f, 1.222f, .232f, 1.232f, .213f, 1.213f, .223f, 1.223f, .233f, 1.233f
                                                 };
    param1 = mod.add_parameters({3});
    TensorTools::set_elements(param1.get_storage().values, param1_vals);
    param2 = mod.add_parameters({3});
    TensorTools::set_elements(param2.get_storage().values, param2_vals);
    param3 = mod.add_parameters({3});
    TensorTools::set_elements(param3.get_storage().values, param3_vals);
    param4 = mod.add_parameters({6});
    TensorTools::set_elements(param4.get_storage().values, param4_vals);
    param5 = mod.add_parameters({3});
    TensorTools::set_elements(param5.get_storage().values, param5_vals);
    param_scalar1 = mod.add_parameters({1});
    TensorTools::set_elements(param_scalar1.get_storage().values, param_scalar1_vals);
    param_scalar2 = mod.add_parameters({1});
    TensorTools::set_elements(param_scalar2.get_storage().values, param_scalar2_vals);
    param_kernel1 = mod.add_parameters({3, 2});
    TensorTools::set_elements(param_kernel1.get_storage().values, param_kernel1_vals);
    param_filter1 = mod.add_parameters({3, 2, 2});
    TensorTools::set_elements(param_filter1.get_storage().values, param_filter1_vals);
    param_square1 = mod.add_parameters({3, 3});
    TensorTools::set_elements(param_square1.get_storage().values, param_square1_vals);
    param_cube1 = mod.add_parameters({3, 3, 3});
    TensorTools::set_elements(param_cube1.get_storage().values, param_cube1_vals);
    param_cube2 = mod.add_parameters({3, 3, 6});
    TensorTools::set_elements(param_cube2.get_storage().values, param_cube2_vals);
    lookup1 = mod.add_lookup_parameters(3, {3});
    TensorTools::set_elements(lookup1.get_storage().all_values, param_square1_vals);
    lookup2 = mod.add_lookup_parameters(10, {3});
    lookup3 = mod2.add_lookup_parameters(10, {3});
    lookup4 = mod.add_lookup_parameters(10, {2,3,4,5});
  }
  ~NodeTest() {
    // for (auto x : av) free(x);
  }

  template <class T>
  std::string print_vec(const std::vector<T> vec) {
    ostringstream oss;
    if (vec.size()) oss << vec[0];
    for (size_t i = 1; i < vec.size(); i++)
      oss << ' ' << vec[i];
    return oss.str();
  }

  // When testing a function that produces a non-scalar result, we need to
  // convert the tensor to a scalar so that we can backprop. However, if you
  // aren't careful, you can end up with partial derivatives that have
  // symmetries that are likely to mask certain bugs. This function provides
  // "asymmetric" gradients onto the tensor valued function.
  static Expression to_scalar(const Expression& e) {
    // square = guarantee element's gradients are not all 1
    // sqrt = if e has multiple batches, guarantees the gradients onto the
    // elements of the batch will be not all 1.
    return sqrt(sum_elems(square(e)));
  }

  std::vector<float> ones3_vals, ones2_vals, first_one_vals, batch_vals;
  std::vector<char*> av;
  dynet::ParameterCollection mod, mod2;
  dynet::Parameter param1, param2, param3, param4, param5, param_scalar1, param_scalar2, param_kernel1, param_filter1, param_square1, param_cube1, param_cube2;
  dynet::LookupParameter lookup1, lookup2, lookup3, lookup4;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(node_test, NodeTest);

// Expression constant(const Dim d, float val);
BOOST_AUTO_TEST_CASE( constant_value ) {
  dynet::ComputationGraph cg;
  float mystery_constant = 3.14159f;
  Expression x = constant(cg, Dim({3}), mystery_constant);
  vector<float> z = as_vector(x.value());
  for (unsigned i = 0; i < 3; i++)
    BOOST_CHECK_EQUAL(z[i], mystery_constant);
}

// Expression zeros(const Dim d, float val);
BOOST_AUTO_TEST_CASE( zeros_value ) {
  dynet::ComputationGraph cg;
  Expression x = zeros(cg, Dim({3}));
  vector<float> z = as_vector(x.value());
  for (unsigned i = 0; i < 3; i++)
    BOOST_CHECK_EQUAL(z[i], 0.f);
}

// Expression operator-(const Expression& x);
BOOST_AUTO_TEST_CASE( negate_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = -x1;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( add_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cadd_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cadd_scalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = (x1 + x2) + (x2 + x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cadd_broadcast2_gradient ) {
  Dim dim_permutations[] = {Dim({3,1},2), Dim({3,2},1)};
  dynet::ComputationGraph cg;
  for(int i=0; i<2; i++){
    Dim dim = dim_permutations[i];
    Expression x1 = reshape(parameter(cg, param1), Dim({3,1},1));
    Expression x2 = reshape(parameter(cg, param4), dim);
    Expression y = (x1 + x2) + (x2 + x1);
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cadd_broadcast3_gradient ) {
  Dim dim_permutations[] = {Dim({3,3,3},1), Dim({3,3,1},3), Dim({1,3,3},3), Dim({9,3,1},1), Dim({1,3,9},1), Dim({1,3,1},9), Dim({3,3},3), Dim({9,3},1), Dim({1,3},9)};
  dynet::ComputationGraph cg;
  for(int i=0; i<6; i++){
    Dim dim = dim_permutations[i];
    Expression x1 = reshape(parameter(cg, param1), Dim({1,3,1},1));
    Expression x2 = reshape(parameter(cg, param_cube1), dim);
    Expression y = (x1 + x2) + (x2 - x1);
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cadd_broadcast2_neg_val ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param1), Dim({3,1},1));
  Expression x2 = reshape(parameter(cg, param4), Dim({3,1},2));
  Expression y = x1 - x2;
  Expression z = sum_batches(sum_elems(y));
  BOOST_CHECK_CLOSE(as_scalar(z.value()), -6.5, 0.001);
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_add_1_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_add_2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_scalar2);
  Expression x2 = parameter(cg, param1);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_add_batch1_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param_square1), Dim({1, 3}, 3));
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = x1 + x2;
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_add_batch2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = reshape(parameter(cg, param_square1), Dim({1}, 9));
  Expression y = x1 + x2;
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_sub_1_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = x1 - x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_sub_2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_scalar2);
  Expression x2 = parameter(cg, param1);
  Expression y = x1 - x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_sub_batch1_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param_square1), Dim({1, 3}, 3));
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = x1 - x2;
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_expr_sub_batch2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = reshape(parameter(cg, param_square1), Dim({1}, 9));
  Expression y = x1 - x2;
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}


// Expression sum(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( sum_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = sum({x2, x1, x2});
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( sum_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression x3 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = sum({x3, x1, cmult(x2, x3)});
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( empty_sum ) {
  dynet::ComputationGraph cg;
  vector<Expression> y;
  BOOST_CHECK_THROW(as_vector(sum(y).value()), std::invalid_argument);
}

// Expression sum(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( cumsum_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_cube1);
  vector<Expression> y;
  for (unsigned d=0;d<3;d++){
      y.push_back(squared_norm(cumsum(x, d)));
  }
  Expression z = sum(y);
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

// Expression logsumexp(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( logsumexp_vector_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = to_scalar(logsumexp({x1, x2}));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logsumexp(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( logsumexp_singleelem_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param1);
  Expression y = reshape(x, Dim({1}, 3));
  Expression z = sum_batches(logsumexp({y}));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logsumexp(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( logsumexp_inequal_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression x3 = x1 + x2;
  Expression z = sum_batches(to_scalar(logsumexp({x1, x3})));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logsumexp(x);
BOOST_AUTO_TEST_CASE( logsumexp_dim_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_square1);
  vector<Expression> exps;
  for (int d = 1; d >= 0; d--)
    exps.push_back(logsumexp_dim(x, d));
  Expression z = to_scalar(sum(exps));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( addscalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1 + 2.0;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator+(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalaradd_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0 + x1;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator-(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( subtract_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1 - x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator-(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalarsubtract_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0 - x1;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator-(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( subtractscalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1 - 2.0;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1 * transpose(x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = x1 * transpose(x2);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = sqrt(affine_transform({x1, x2, scalar}));
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
  BOOST_CHECK(y.dim() == x1.dim());
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = sqrt(affine_transform({x1, x2, scalar}));
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch_col_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = input(cg, Dim({1, 3}, 2), batch_vals);
  Expression y = sqrt(affine_transform({transpose(x1), scalar, x2}));
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = input(cg, Dim({1, 3}, 2), batch_vals);
  Expression scalar = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = sqrt( affine_transform({x1, scalar, transpose(x2) }) );
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( affine_batch3_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_square1);
  Expression inp = input(cg, Dim({3}, 2), batch_vals);
  Expression y = sqrt( affine_transform({x1, x2, inp }) );
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression operator*(const Expression& x, float y);
BOOST_AUTO_TEST_CASE( multiplyscalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1 * 2.0;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// inline Expression operator*(float y, const Expression& x) { return x * y; }
BOOST_AUTO_TEST_CASE( scalarmultiply_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0 * x1;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }
BOOST_AUTO_TEST_CASE( dividescalar_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1 / 2.0;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cdiv(x1, x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = cdiv(x2, x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_cdiv_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = cdiv(x1, x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_cdiv_batch1_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param_square1), Dim({1, 3}, 3));
  Expression x2 = parameter(cg, param_scalar2);
  Expression y = cdiv(x1, x2);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_cdiv_batch2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = cdiv(x2, x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_broadcast2_gradient ) {
  Dim dim_permutations[] = {Dim({3,1},2), Dim({3,2},1)};
  dynet::ComputationGraph cg;
  for(int i=0; i<2; i++){
    Dim dim = dim_permutations[i];
    Expression x1 = reshape(parameter(cg, param1), Dim({3,1},1));
    Expression x2 = reshape(parameter(cg, param4), dim);
    Expression y = cdiv(x2, x1);
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_broadcast3_gradient ) {
  Dim dim_permutations[] = {Dim({3,3,3},1), Dim({3,3,1},3), Dim({1,3,3},3), Dim({9,3,1},1), Dim({1,3,9},1), Dim({1,3,1},9)};
  dynet::ComputationGraph cg;
  for(int i=0; i<6; i++){
    Dim dim = dim_permutations[i];
    Expression x1 = reshape(parameter(cg, param1), Dim({1,3,1},1));
    Expression x2 = reshape(parameter(cg, param_cube1), dim);
    Expression y = cdiv(x2, x1);
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}


// Expression colwise_add(const Expression& x, const Expression& bias);
BOOST_AUTO_TEST_CASE( colwise_add_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = colwise_add(x1 * transpose(x2), x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression colwise_add(const Expression& x, const Expression& bias);
BOOST_AUTO_TEST_CASE( colwise_add_batch1_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression x3 = input(cg, Dim({1, 3}, 2), batch_vals);
  Expression y = colwise_add(x1 * x3, x2);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression colwise_add(const Expression& x, const Expression& bias);
BOOST_AUTO_TEST_CASE( colwise_add_batch2_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression x3 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = colwise_add(x1 * transpose(x2), cmult(x2, x3));
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate_cols(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_cols_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = concatenate_cols({x1, x2, x1});
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate_cols(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_cols_vecmatrix_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_square1);
  Expression y = concatenate_cols({x1, x2, x1});
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate_to_batch(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_to_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression xsquare = parameter(cg, param_square1);
  Expression y = concatenate_to_batch({x1, x2});
  Expression z = sum_batches(to_scalar(xsquare * y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = transpose(parameter(cg, param1));
  Expression x2 = transpose(parameter(cg, param2));
  Expression y = concatenate({x1, x2, x1});
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression concatenate(const std::initializer_list<Expression>& xs);
BOOST_AUTO_TEST_CASE( concatenate_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = transpose(parameter(cg, param1));
  Expression x2 = transpose(parameter(cg, param2));
  Expression x3 = input(cg, Dim({1, 3}, 2), batch_vals);
  Expression y = concatenate({x1, x2, cmult(x2, x3)});
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
BOOST_AUTO_TEST_CASE( contract3d_1d_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression square1 = parameter(cg, param_square1);
  Expression cube1 = parameter(cg, param_cube1);
  Expression y = contract3d_1d(cube1, x1, square1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
BOOST_AUTO_TEST_CASE( contract3d_batch_1d_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression square1 = parameter(cg, param_square1);
  Expression cube1 = parameter(cg, param_cube1);
  Expression batched_cube1 = concatenate_to_batch({cube1, cube1, cube1});
  Expression y = contract3d_1d(batched_cube1, x1, square1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
BOOST_AUTO_TEST_CASE( contract3d_1d_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression batched_x1 = reshape(parameter(cg, param_square1), Dim({3}, 3));
  Expression square1 = parameter(cg, param_square1);
  Expression cube1 = parameter(cg, param_cube1);
  Expression y = contract3d_1d(cube1, batched_x1, square1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
BOOST_AUTO_TEST_CASE( contract3d_batch_1d_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression batched_x1 = reshape(parameter(cg, param_square1), Dim({3}, 3));
  Expression square1 = parameter(cg, param_square1);
  Expression cube1 = parameter(cg, param_cube1);
  Expression batched_cube1 = concatenate_to_batch({cube1, cube1, cube1});
  Expression y = contract3d_1d(batched_cube1, batched_x1, square1);
  Expression z = sum_batches(to_scalar(y));
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
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sqrt(const Expression& x);
BOOST_AUTO_TEST_CASE( sqrt_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = sqrt(x3);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression abs(const Expression& x);
BOOST_AUTO_TEST_CASE( abs_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = abs(x3);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression erf(const Expression& x);
BOOST_AUTO_TEST_CASE( erf_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = erf(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sin(const Expression& x);
BOOST_AUTO_TEST_CASE( sin_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = sin(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cos(const Expression& x);
BOOST_AUTO_TEST_CASE( cos_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = cos(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression tan(const Expression& x);
BOOST_AUTO_TEST_CASE( tan_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = tan(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression asin(const Expression& x);
BOOST_AUTO_TEST_CASE( asin_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param5);
  Expression y = asin(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression acos(const Expression& x);
BOOST_AUTO_TEST_CASE( acos_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param5);
  Expression y = acos(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression atan(const Expression& x);
BOOST_AUTO_TEST_CASE( atan_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param5);
  Expression y = atan(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sinh(const Expression& x);
BOOST_AUTO_TEST_CASE( sinh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = sinh(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cosh(const Expression& x);
BOOST_AUTO_TEST_CASE( cosh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = cosh(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression tanh(const Expression& x);
BOOST_AUTO_TEST_CASE( tanh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = tanh(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression asinh(const Expression& x);
BOOST_AUTO_TEST_CASE( asinh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param3);
  Expression y = asinh(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression acosh(const Expression& x);
BOOST_AUTO_TEST_CASE( acosh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param3);
  Expression y = acosh(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression atanh(const Expression& x);
BOOST_AUTO_TEST_CASE( atanh_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param5);
  Expression y = atanh(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression exp(const Expression& x);
BOOST_AUTO_TEST_CASE( exp_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = exp(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression square(const Expression& x);
BOOST_AUTO_TEST_CASE( square_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = square(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cube(const Expression& x);
BOOST_AUTO_TEST_CASE( cube_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = cube(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_sigmoid(const Expression& x);
BOOST_AUTO_TEST_CASE( log_sigmoid_gradient ) {
  dynet::ComputationGraph cg;
  Expression x2 = parameter(cg, param2);
  Expression y = log_sigmoid(x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression lgamma(const Expression& x);
BOOST_AUTO_TEST_CASE( lgamma_gradient ) {
  dynet::ComputationGraph cg;
  Expression x2 = parameter(cg, param2);
  Expression y = lgamma(x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log(const Expression& x);
BOOST_AUTO_TEST_CASE( log_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = log(x3);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logistic(const Expression& x);
BOOST_AUTO_TEST_CASE( logistic_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = logistic(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression rectify(const Expression& x);
BOOST_AUTO_TEST_CASE( rectify_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = rectify(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}


// Expression elu(const Expression& x);
BOOST_AUTO_TEST_CASE( elu_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = elu(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression selu(const Expression& x);
BOOST_AUTO_TEST_CASE( selu_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = selu(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression silu(const Expression& x);
BOOST_AUTO_TEST_CASE( silu_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = silu(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression round(const Expression& x, GradientMode gradient_mode);
BOOST_AUTO_TEST_CASE( round_forward ) {
  // batch_vals = {1.1f, 2.6f, 3.3f, 4.0f, 5.1f, 6.6f};
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression y = round(x, zero_gradient);
  std::vector<float> v = as_vector(y.value());
  BOOST_CHECK_EQUAL(v[0], 1.0);
  BOOST_CHECK_EQUAL(v[1], 3.0);
  BOOST_CHECK_EQUAL(v[2], 3.0);
  BOOST_CHECK_EQUAL(v[3], 4.0);
  BOOST_CHECK_EQUAL(v[4], 5.0);
  BOOST_CHECK_EQUAL(v[5], 7.0);
}

// Expression ceil(const Expression& x, GradientMode gradient_mode);
BOOST_AUTO_TEST_CASE( ceil_forward ) {
  // batch_vals = {1.1f, 2.6f, 3.3f, 4.0f, 5.1f, 6.6f};
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression y = ceil(x, zero_gradient);
  std::vector<float> v = as_vector(y.value());
  BOOST_CHECK_EQUAL(v[0], 2.0);
  BOOST_CHECK_EQUAL(v[1], 3.0);
  BOOST_CHECK_EQUAL(v[2], 4.0);
  BOOST_CHECK_EQUAL(v[3], 4.0);
  BOOST_CHECK_EQUAL(v[4], 6.0);
  BOOST_CHECK_EQUAL(v[5], 7.0);
}

// Expression floor(const Expression& x, GradientMode gradient_mode);
BOOST_AUTO_TEST_CASE( floor_forward ) {
  // batch_vals = {1.1f, 2.6f, 3.3f, 4.0f, 5.1f, 6.6f};
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression y = floor(x, zero_gradient);
  std::vector<float> v = as_vector(y.value());
  BOOST_CHECK_EQUAL(v[0], 1.0);
  BOOST_CHECK_EQUAL(v[1], 2.0);
  BOOST_CHECK_EQUAL(v[2], 3.0);
  BOOST_CHECK_EQUAL(v[3], 4.0);
  BOOST_CHECK_EQUAL(v[4], 5.0);
  BOOST_CHECK_EQUAL(v[5], 6.0);
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
BOOST_AUTO_TEST_CASE( hinge_multiple_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  vector<Expression> exp;
  for (unsigned index = 3; index > 0; --index)
    exp.push_back(hinge(x1, index - 1, 0.5));
  Expression z = sum(exp);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression hinge(const Expression& x, unsigned index, float m = 1.0);
BOOST_AUTO_TEST_CASE( hinge_batch_gradient ) {
  std::vector<unsigned> idx = {1, 2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(hinge(x1 + x2, idx, 2.f));
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

// Expression hinge_dim(const Expression& x, unsigned index, unsigned dim = 0, float m = 1.0);
BOOST_AUTO_TEST_CASE( hinge_dim_gradient ) {
  std::vector<unsigned> index = {0, 1, 2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_square1);
  Expression z = to_scalar(hinge_dim(x1, index, 0, 0.5) + hinge_dim(x1, index, 1, 0.5));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( log_softmax_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log_softmax(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( log_softmax_autobatch_gradient ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 1;
  dynet::ComputationGraph cg;
  vector<Expression> vals;
  {
    Expression x1 = parameter(cg, param1);
    vals.push_back(log_softmax(x1));
  }
  {
    Expression x2 = parameter(cg, param2);
    vals.push_back(log_softmax(x2));
  }
  Expression y = sum(vals);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
  dynet::autobatch_flag = autobatch_cache;
}

// Expression log_softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( log_softmax_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = log_softmax(x1 + x2);
  Expression z = sum_batches(input(cg, {1, 3}, first_one_vals) * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( log_softmax_colbatch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = reshape(parameter(cg, param_cube1), Dim({3, 3}, 3));
  Expression y = log_softmax(x);
  Expression z = sum_batches(input(cg, {1, 3}, first_one_vals) * y * input(cg, {3}, first_one_vals));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);
BOOST_AUTO_TEST_CASE( restricted_log_softmax_gradient ) {
  vector<unsigned> restriction = {0, 1};
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression y = exp( log_softmax(x3, restriction) );
  Expression z = input(cg, {1, 3}, first_one_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( softmax_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log(softmax(x1));
  Expression z = input(cg, {1, 3}, first_one_vals) * y;
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( softmax_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = log(softmax(x1 + x2));
  Expression z = sum_batches(input(cg, {1, 3}, first_one_vals) * y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( softmax_colbatch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = reshape(parameter(cg, param_cube1), Dim({3, 3}, 3));
  Expression y = softmax(x);
  Expression z = sum_batches(input(cg, {1, 3}, first_one_vals) * y * input(cg, {3}, first_one_vals));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression softmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( softmax_cols_colbatch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = reshape(parameter(cg, param_cube1), Dim({3, 3}, 3));
  Expression y = softmax(x, 1);
  Expression z = sum_batches(input(cg, {1, 3}, first_one_vals) * y * input(cg, {3}, first_one_vals));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sparsemax(const Expression& x);
BOOST_AUTO_TEST_CASE( sparsemax_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = sparsemax(x1);
  Expression z = input(cg, {1, 3}, first_one_vals) * y;
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
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pow(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( pow_gradient ) {
  dynet::ComputationGraph cg;
  Expression x3 = parameter(cg, param3);
  Expression x_scalar1 = parameter(cg, param_scalar1);
  Expression y = pow(x3, x_scalar1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression min(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( min_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = min(x1, x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression max(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( max_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = max(x1, x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// TODO: Noise is random, so it cannot be tested simply?
// Expression noise(const Expression& x, real stddev);
BOOST_AUTO_TEST_CASE( noise_forward ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = noise(x1, 0.5);
  Expression z = to_scalar(y);
  cg.forward(z);
}

//TODO: Dropout scales the gradients at training time, so they don't match.
// Expression dropout(const Expression& x, real p);
BOOST_AUTO_TEST_CASE( dropout_forward ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = dropout(x1, 0.5);
  Expression z = to_scalar(y);
  cg.forward(z);
}

BOOST_AUTO_TEST_CASE( dropout_batch_forward ) {
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression y = dropout_batch(x, 0.5);
  Expression z = to_scalar(y);
  cg.forward(z);
}


BOOST_AUTO_TEST_CASE( dropout_dim_forward ) {
  for (unsigned d = 0; d < 3; d++) {
    dynet::ComputationGraph cg;
    Expression x = parameter(cg, param_cube1);
    Expression y = dropout_dim(x, d, 0.5);
    Expression z = to_scalar(y);
    cg.forward(z);
  }
}

// TODO: Dropout scales the gradients at training time, so they don't match.
// Expression block_dropout(const Expression& x, real p);

// Expression argmax(const Expression& x, GradientMode gradient_mode);
BOOST_AUTO_TEST_CASE( argmax_forward ) {
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression y = argmax(x, zero_gradient);
  std::vector<float> v = as_vector(y.value());
  BOOST_CHECK_EQUAL(v[0], 0.0);
  BOOST_CHECK_EQUAL(v[1], 0.0);
  BOOST_CHECK_EQUAL(v[2], 1.0);
  BOOST_CHECK_EQUAL(v[3], 0.0);
  BOOST_CHECK_EQUAL(v[4], 0.0);
  BOOST_CHECK_EQUAL(v[5], 1.0);
}

// Expression argmax(const Expression& x, GradientMode gradient_mode);
BOOST_AUTO_TEST_CASE( argmax_backward ) {
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression y = argmax(x, zero_gradient);
  Expression z = sum_batches(squared_norm(y));
  cg.backward(z, true);
  std::vector<float> g_x = as_vector(x.gradient());
  BOOST_CHECK_EQUAL(g_x[0], 0.0);
  BOOST_CHECK_EQUAL(g_x[1], 0.0);
  BOOST_CHECK_EQUAL(g_x[2], 0.0);
  BOOST_CHECK_EQUAL(g_x[3], 0.0);
  BOOST_CHECK_EQUAL(g_x[4], 0.0);
  BOOST_CHECK_EQUAL(g_x[5], 0.0);
}

// Expression argmax(const Expression& x, GradientMode gradient_mode);
BOOST_AUTO_TEST_CASE( straight_through_backward ) {
  dynet::ComputationGraph cg;
  Expression x = input(cg, Dim({3}, 2), batch_vals);
  Expression x_ = input(cg, Dim({3}, 2), batch_vals);
  Expression y = argmax(x, straight_through_gradient);
  Expression z = sum_batches(dot_product(y, x_));
  cg.backward(z, true);
  std::vector<float> g_x = as_vector(x.gradient());
  BOOST_CHECK_EQUAL(g_x[0], batch_vals[0]);
  BOOST_CHECK_EQUAL(g_x[1], batch_vals[1]);
  BOOST_CHECK_EQUAL(g_x[2], batch_vals[2]);
  BOOST_CHECK_EQUAL(g_x[3], batch_vals[3]);
  BOOST_CHECK_EQUAL(g_x[4], batch_vals[4]);
  BOOST_CHECK_EQUAL(g_x[5], batch_vals[5]);
}

// Expression reshape(const Expression& x, const Dim& d);
BOOST_AUTO_TEST_CASE( reshape_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = reshape(x1, {1, 3});
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression reshape(const Expression& x, const Dim& d);
BOOST_AUTO_TEST_CASE( reshape_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y1 = x1 * transpose(x2);
  Expression y2 = reshape(y1, Dim({3, 3}, 2));
  Expression z = sum_batches(to_scalar(y2));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression transpose(const Expression& x);
BOOST_AUTO_TEST_CASE( transpose_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_square1);
  Expression y = x1 * transpose(x1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression transpose(const Expression& x);
BOOST_AUTO_TEST_CASE( transpose_higherorder_gradient ) {
  dynet::ComputationGraph cg;
  Expression cube1 = parameter(cg, param_cube1);
  Expression x1 = reshape(transpose(cube1, {2, 0, 1}), Dim({9, 3}));
  Expression x2 = reshape(transpose(cube1, {1, 2, 0}), Dim({3, 9}));
  Expression z = to_scalar(x1 * x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression logdet(const Expression& x);
BOOST_AUTO_TEST_CASE( logdet_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_square1);
  Expression y = logdet(-x);
  BOOST_CHECK(check_grad(mod, y, 0));
}

// Expression inverse(const Expression& x);
BOOST_AUTO_TEST_CASE( inverse_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_square1);
  Expression y = inverse(x);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression trace_of_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( trace_of_product_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression z = trace_of_product(x1, x2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cadd_broadcast_gradient_scalar ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param_scalar1), Dim({1},1));
  Expression x2 = reshape(parameter(cg, param4), Dim({3,1,1},2));
  Expression y = (x1 + x2) + (x2 + x1) + (x1 - x2) + (x2 - x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}
// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_broadcast_gradient_scalar ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param_scalar1), Dim({1},1));
  Expression x2 = reshape(parameter(cg, param4), Dim({3,1,1},2));
  Expression y = cdiv(x2, x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cmult(x1, x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression y = cmult(x1, x2) + cmult(x2, x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_cmult_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_scalar1);
  Expression x2 = parameter(cg, param2);
  Expression y = cmult(x1, x2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalar_cmult_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_scalar1);
  Expression x2 = reshape(parameter(cg, param_square1), Dim({1, 3}, 3));
  Expression y = cmult(x1, x2) + cmult(x2, x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_broadcast_gradient_scalar ) {
  dynet::ComputationGraph cg;
  Expression x1 = reshape(parameter(cg, param_scalar1), Dim({1},1));
  Expression x2 = reshape(parameter(cg, param4), Dim({3,1,1},2));
  Expression y = cmult(x1, x2) + cmult(x2, x1);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_broadcast2_gradient ) {
  Dim dim_permutations[] = {Dim({3,1},2), Dim({3,2},1)};
  dynet::ComputationGraph cg;
  for(int i=0; i<2; i++){
    Dim dim = dim_permutations[i];
    Expression x1 = reshape(parameter(cg, param1), Dim({3,1},1));
    Expression x2 = reshape(parameter(cg, param4), dim);
    Expression y = cmult(x1, x2) + cmult(x2, x1);
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression cmult(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cmult_broadcast3_gradient ) {
  Dim dim_permutations[] = {Dim({3,3,3},1), Dim({3,3,1},3), Dim({1,3,3},3), Dim({9,3,1},1), Dim({1,3,9},1), Dim({1,3,1},9)};
  dynet::ComputationGraph cg;
  for(int i=0; i<6; i++){
    Dim dim = dim_permutations[i];
    Expression x1 = reshape(parameter(cg, param1), Dim({1,3,1},1));
    Expression x2 = reshape(parameter(cg, param_cube1), dim);
    Expression y = cmult(x1, x2) + cmult(x2, x1);
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
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
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(dot_product(x1, x2) + dot_product(x2, x1) * 2);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression dot_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( dot_product_matrix_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_square1);
  Expression z = dot_product(x1, x1);
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

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_distance_batchright_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(squared_distance(x1, x1 + x2));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_distance_batchleft_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(squared_distance(x1 + x2, x1));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_distance_batchboth_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(squared_distance(x1 + x2, cmult(x1, x2)));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_norm_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = squared_norm(x1);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_norm_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(squared_norm(x1 + x2));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression l2_norm(const Expression&);
BOOST_AUTO_TEST_CASE( l2_norm_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression z = l2_norm(x1);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression l2_norm(const Expression& x);
BOOST_AUTO_TEST_CASE( l2_norm_batch_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(l2_norm(x1 + x2));
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

// Expression binary_log_loss(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( binary_log_loss_edgecases ) {
  dynet::ComputationGraph cg;
  float val, infinity = - log(DYNET_DEVICE_MIN);
  Expression x, y, z;
  vector<float> values = {0.0, 0.5, 1.0};
  for (float vx : values) {
    for (float vy : values) {
      x = input(cg, vx);
      // and y == 0
      y = input(cg, vy);
      z = binary_log_loss(x, y);
      val = as_scalar(z.value());
      if (vx == 0.5)
        BOOST_CHECK_CLOSE(val, log(2), 0.1);
      else if (vx == vy)
        BOOST_CHECK_CLOSE(val, 0, 0.1);
      else
        BOOST_CHECK_CLOSE(val, infinity, 0.1);
    }
  }

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

/*
// Expression conv1d_narrow(const Expression& x, const Expression& f);
BOOST_AUTO_TEST_CASE( conv1d_narrow_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression xkernel = parameter(cg, param_kernel1);
  Expression y = conv1d_narrow(xsquare, xkernel);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression conv1d_wide(const Expression& x, const Expression& f);
BOOST_AUTO_TEST_CASE( conv1d_wide_gradient ) {
  dynet::ComputationGraph cg;
  Expression xkernel = parameter(cg, param_kernel1);
  Expression y = conv1d_wide(xkernel, xkernel);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}
*/

// Expression filter1d_narrow(const Expression& x, const Expression& f);
BOOST_AUTO_TEST_CASE( filter1d_narrow_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression xfilter = parameter(cg, param_filter1);
  Expression y = filter1d_narrow(xsquare, xfilter);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression kmax_pooling(const Expression& x, unsigned k);
BOOST_AUTO_TEST_CASE( kmax_pooling_keq1_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(kmax_pooling(xsquare, 1));
  Expression z = pickneglogsoftmax(y, 1);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression kmax_pooling(const Expression& x, unsigned k);
BOOST_AUTO_TEST_CASE( kmax_pooling_keq2_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(kmax_pooling(xsquare, 2));
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression fold_rows(const Expression& x, unsigned nrows=2);
BOOST_AUTO_TEST_CASE( fold_rows_gradient ) {
  dynet::ComputationGraph cg;
  Expression x4 = parameter(cg, param4);
  Expression y = fold_rows(x4, 2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression circ_corr(const Expression& u, const Expression& v);
BOOST_AUTO_TEST_CASE( circ_corr_gradient ) {
  dynet::ComputationGraph cg;
  Expression u = parameter(cg, param5);
  Expression v = parameter(cg, param2);
  Expression y = circ_corr(u, v);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression circ_conv(const Expression& u, const Expression& v);
BOOST_AUTO_TEST_CASE( circ_conv_gradient ) {
  dynet::ComputationGraph cg;
  Expression u = parameter(cg, param5);
  Expression v = parameter(cg, param2);
  Expression y = circ_conv(u, v);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression average(const Expression& x);
BOOST_AUTO_TEST_CASE( average_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression x3 = parameter(cg, param3);
  Expression y = average({x1, x2, x3});
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression average_cols(const Expression& x);
BOOST_AUTO_TEST_CASE( average_cols_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(average_cols(xsquare));
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum_cols(const Expression& x);
BOOST_AUTO_TEST_CASE( sum_cols_gradient ) {
  dynet::ComputationGraph cg;
  Expression xsquare = parameter(cg, param_square1);
  Expression y = tanh(sum_cols(xsquare));
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression conv2d(const Expression& x ,const Expression& f, const std::vector<unsigned>& stride, bool is_valid);
BOOST_AUTO_TEST_CASE( conv2d_valid_gradient ) {
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 2, 2, 3});
  std::vector<float> param_kernel_vals = {.011f, .022f, .033f, .012f, .022f, .032f, .013f, .023f, .033f,
                                          .111f, -.122f, -.033f, -.112f, -.022f, -.132f, -.113f, -.123f, -.133f,
                                          .211f, .222f, .233f, .212f, .222f, .232f
                                         };
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  std::vector<float> conv2d_batch_vals(50 * 50 * 2 * 2);
  for (unsigned i = 0; i < conv2d_batch_vals.size(); ++i) {
    conv2d_batch_vals[i] = i * 0.011f + (i + 1) * 0.001f;
  }
  Expression x = input(cg, Dim({50, 50, 2}, 2), conv2d_batch_vals);
  Expression kernel = parameter(cg, param_kernel);
  vector<unsigned> stride = {3, 3}; bool is_valid = true;
  Expression y = conv2d(x, kernel, stride, is_valid);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression log_softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( conv2d_autobatch_gradient ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 1;
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 2, 2, 3});
  std::vector<float> param_kernel_vals = {.011f, .022f, .033f, .012f, .022f, .032f, .013f, .023f, .033f,
                                          .111f, -.122f, -.033f, -.112f, -.022f, -.132f, -.113f, -.123f, -.133f,
                                          .211f, .222f, .233f, .212f, .222f, .232f
                                         };
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  Expression kernel = parameter(cg, param_kernel);
  vector<unsigned> stride = {3, 3}; bool is_valid = true;
  std::vector<float> conv2d_vals1(50 * 50 * 2), conv2d_vals2(50 * 50 * 2);
  for (unsigned i = 0; i < conv2d_vals1.size(); ++i) {
    conv2d_vals1[i] = i * 0.011f + (i + 1) * 0.001f;
    conv2d_vals2[i] = i * 0.015f + (i + 1) * -0.001f;
  }
  vector<Expression> zs;
  {
    Expression x = input(cg, Dim({50, 50, 2}), conv2d_vals1);
    Expression y = conv2d(x, kernel, stride, is_valid);
    zs.push_back(to_scalar(y));
  }
  {
    Expression x = input(cg, Dim({50, 50, 2}), conv2d_vals2);
    Expression y = conv2d(x, kernel, stride, is_valid);
    zs.push_back(to_scalar(y));
  }
  Expression z = sum(zs);
  BOOST_CHECK(check_grad(mod, z, 0));
  dynet::autobatch_flag = autobatch_cache;
}

// Expression conv2d(const Expression& x ,const Expression& f, const std::vector<unsigned>& stride, bool is_valid);
BOOST_AUTO_TEST_CASE( conv2d_valid_singlefilter_gradient ) {
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 4, 1, 3});
  std::vector<float> param_kernel_vals = {.011f, .022f, .033f, .012f, .022f, .032f, .013f, .023f, .033f,
                                          .111f, -.122f, -.033f, -.112f, -.022f, -.132f, -.113f, -.123f, -.133f,
                                          .211f, .222f, .233f, .212f, .222f, .232f
                                         };
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  std::vector<float> conv2d_batch_vals(50 * 100 * 1 * 2);
  for (unsigned i = 0; i < conv2d_batch_vals.size(); ++i) {
    conv2d_batch_vals[i] = i * 0.011f + (i + 1) * 0.001f;
  }
  Expression x = input(cg, Dim({50, 100}, 2), conv2d_batch_vals);
  Expression kernel = parameter(cg, param_kernel);
  vector<unsigned> stride = {3, 3}; bool is_valid = true;
  Expression y = conv2d(x, kernel, stride, is_valid);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( conv2d_same_gradient ) {
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 2, 2, 3});
  std::vector<float> param_kernel_vals = {.011f, .022f, .033f, .012f, .022f, .032f, .013f, .023f, .033f,
                                          .111f, -.122f, -.033f, -.112f, -.022f, -.132f, -.113f, -.123f, -.133f,
                                          .211f, .222f, .233f, .212f, .222f, .232f
                                         };
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  Parameter param_kernel2 = mod.add_parameters({2, 2, 3, 2});
  TensorTools::set_elements(param_kernel2.get_storage().values, param_kernel_vals);

  std::vector<float> conv2d_batch_vals(2 * 50 * 50 * 2);
  for (unsigned i = 0; i < conv2d_batch_vals.size(); ++i) {
    conv2d_batch_vals[i] = i * 0.011f + (i + 1) * 0.001f;
  }
  Expression x = input(cg, Dim({50, 50, 2}, 2), conv2d_batch_vals);
  Expression kernel = parameter(cg, param_kernel);
  vector<unsigned> stride = {4, 4}; bool is_valid = false;
  Expression y = conv2d(x, kernel, stride, is_valid);
  Expression kernel2 = parameter(cg, param_kernel2);
  Expression y2 = conv2d(y, kernel2, stride, is_valid);
  Expression z = sum_batches(to_scalar(y2));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( maxpooling2d_same_gradient ) {
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 2, 1, 1});
  std::vector<float> param_kernel_vals = {.011f, .022f, .012f, .022f};
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  std::vector<float> maxpooling2d_batch_vals(1 * 11 * 11 * 2);
  for (unsigned i = 0; i < maxpooling2d_batch_vals.size(); ++i) {
    maxpooling2d_batch_vals[i] = i * 0.011f + (i + 1) * 0.001f;
  }
  Expression x = input(cg, Dim({11, 11, 1}, 2), maxpooling2d_batch_vals);
  Expression kernel = parameter(cg, param_kernel);
  std::vector<unsigned> ksize = {2, 2};
  std::vector<unsigned> stride = {2, 5};
  bool is_valid = false;
  Expression w = conv2d(x, kernel, stride, is_valid);
  //Expression z = sum_batches(to_scalar(w));
  //BOOST_CHECK(check_grad(mod, z, 0));
  is_valid = false;
  Expression y = maxpooling2d(w, ksize, stride, is_valid);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( maxpooling2d_valid_gradient ) {
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 2, 1, 1});
  std::vector<float> param_kernel_vals = {.011f, .022f, .012f, .022f};
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  std::vector<float> maxpooling2d_batch_vals(1 * 21 * 21 * 2);
  for (unsigned i = 0; i < maxpooling2d_batch_vals.size(); ++i) {
    maxpooling2d_batch_vals[i] = i * 0.011f + (i + 1) * 0.001f;
  }
  Expression x = input(cg, Dim({21, 21, 1}, 2), maxpooling2d_batch_vals);
  Expression kernel = parameter(cg, param_kernel);
  std::vector<unsigned> ksize = {2, 2};
  std::vector<unsigned> stride = {2, 5};
  bool is_valid = false;
  Expression w = conv2d(x, kernel, stride, is_valid);
  is_valid = true;
  Expression y = maxpooling2d(w, ksize, stride, is_valid);
  Expression z = sum_batches(to_scalar(y));
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( maxpooling2d_same_gradient_two ) {
  dynet::ComputationGraph cg;
  Parameter param_kernel = mod.add_parameters({2, 2, 1, 1});
  std::vector<float> param_kernel_vals = {.011f, .022f, .012f, .022f};
  TensorTools::set_elements(param_kernel.get_storage().values, param_kernel_vals);
  std::vector<float> maxpooling2d_batch_vals(1 * 31 * 16 * 2);
  for (unsigned i = 0; i < maxpooling2d_batch_vals.size(); ++i) {
    maxpooling2d_batch_vals[i] = i * 0.011f + (i + 1) * 0.001f;
  }
  Expression x = input(cg, Dim({31, 16, 1}, 2), maxpooling2d_batch_vals);
  Expression kernel = parameter(cg, param_kernel);
  std::vector<unsigned> ksize = {3, 2};
  std::vector<unsigned> stride = {3, 3};
  bool is_valid = false;
  Expression w = conv2d(x, kernel, stride, is_valid);
  is_valid = true;
  Expression y = maxpooling2d(w, ksize, stride, is_valid);
  Expression z = sum_batches(to_scalar(y));
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
  std::vector<unsigned> idx = {1, 2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(pick(x1 + x2, idx));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pick_batch_broadcast_gradient ) {
  std::vector<unsigned> idx = {1, 2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_square1);
  Expression z = sum_batches(squared_norm(pick(x1, idx, 0)));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick_batch_elem(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pick_batch_elem_gradient ) {
  unsigned idx = 0;
  dynet::ComputationGraph cg;
  Expression x1 = input(cg, Dim({ 3 }, 2), batch_vals);
  Expression z = sum_rows(pick_batch_elem(x1, idx));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick_batch_elems(const Expression& x, cosnt std::vector<unsigned> & v);
BOOST_AUTO_TEST_CASE(  pick_batch_elems_gradient ) {
  dynet::ComputationGraph cg;
  std::vector<unsigned> indices = { 0, 1 };
  Expression x1 = input(cg, Dim({ 3 }, 2), batch_vals);
  Expression picked_x1 = pick_batch_elems(x1, indices);
  Expression z = sum({
    sum_rows(pick_batch_elem(picked_x1, (unsigned) 0)),
    sum_rows(pick_batch_elem(picked_x1, (unsigned) 1))
  });
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick_range(const Expression& x, unsigned v, unsigned u);
BOOST_AUTO_TEST_CASE( pick_range_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = pick_range(x1, 0, 2);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression pick_range(const Expression& x, unsigned v, unsigned u);
BOOST_AUTO_TEST_CASE( pick_range_dim_gradient ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_square1);
  Expression y = pick_range(x1, 0, 2, 1);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression select_rows(const Expression& x, vector<unsigned>& rows);
BOOST_AUTO_TEST_CASE( select_rows_gradient ) {
  dynet::ComputationGraph cg;
  vector<unsigned> rows = {1};
  Expression x1 = parameter(cg, param_square1);
  Expression y = select_rows(x1, rows);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression select_rows(const Expression& x, vector<unsigned>& rows);
BOOST_AUTO_TEST_CASE( select_rows_multiple_gradient ) {
  dynet::ComputationGraph cg;
  vector<unsigned> rows = {0, 2};
  Expression x1 = parameter(cg, param_square1);
  Expression y = select_rows(x1, rows) * x1;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression select_rows(const Expression& x, vector<unsigned>& rows);
BOOST_AUTO_TEST_CASE( select_rows_oob ) {
  dynet::ComputationGraph cg;
  vector<unsigned> rows = {3};
  Expression x1 = parameter(cg, param_square1);
  Expression y = select_rows(x1, rows);
  BOOST_CHECK_THROW(y.value(), std::invalid_argument);
}

// Expression select_rows(const Expression& x, vector<unsigned>& rows);
BOOST_AUTO_TEST_CASE( select_rows_autobatch_gradient ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 1;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param_square1);
  vector<Expression> vals;
  {
    vector<unsigned> rows = {0, 2};
    Expression y = select_rows(x1, rows) * x1;
    vals.push_back(to_scalar(y));
  }
  {
    vector<unsigned> rows = {2, 1};
    Expression y = select_rows(x1, rows) * x1;
    vals.push_back(to_scalar(y));
  }
  {
    vector<unsigned> rows = {0};
    Expression y = select_rows(x1, rows) * x1;
    vals.push_back(to_scalar(y));
  }
  Expression z = sum(vals);
  BOOST_CHECK(check_grad(mod, z, 0));
  dynet::autobatch_flag = autobatch_cache;
}

// Expression select_cols(const Expression& x, vector<unsigned>& rows);
BOOST_AUTO_TEST_CASE( select_cols_gradient ) {
  dynet::ComputationGraph cg;
  vector<unsigned> cols = {1};
  Expression x1 = parameter(cg, param_square1);
  Expression y = select_cols(x1, cols);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression select_cols(const Expression& x, vector<unsigned>& cols);
BOOST_AUTO_TEST_CASE( select_cols_multiple_gradient ) {
  dynet::ComputationGraph cg;
  vector<unsigned> cols = {0, 2};
  Expression x1 = parameter(cg, param_square1);
  Expression y = x1 * select_cols(x1, cols);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression select_cols(const Expression& x, vector<unsigned>& rows);
BOOST_AUTO_TEST_CASE( select_cols_oob ) {
  dynet::ComputationGraph cg;
  vector<unsigned> cols = {3};
  Expression x1 = parameter(cg, param_square1);
  Expression y = select_cols(x1, cols);
  BOOST_CHECK_THROW(y.value(), std::invalid_argument);
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
  std::vector<unsigned> idx = {1, 2};
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = input(cg, Dim({3}, 2), batch_vals);
  Expression z = sum_batches(pickneglogsoftmax(x1 + x2, idx));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression strided_select(const Expression& x, vector<unsigned>& indices);
BOOST_AUTO_TEST_CASE( strided_select_gradient_noop ) {
  dynet::ComputationGraph cg;
  const vector<int> strides = {};
  Expression x1 = parameter(cg, param_square1);
  Expression y = strided_select(x1, strides);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
  BOOST_CHECK(x1.dim().size() == y.dim().size());
}

// Expression strided_select(const Expression& x, vector<unsigned>& indices);
BOOST_AUTO_TEST_CASE( strided_select_gradient ) {
  dynet::ComputationGraph cg;
  for(int stride=1;stride<4;stride++){
    const vector<int> strides = {stride,stride,stride};
    Expression x1 = parameter(cg, param_cube1);
    Expression y3 = strided_select(x1, strides);
    Expression z = to_scalar(y3);
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}
// Expression strided_select(const Expression& x, vector<unsigned>& indices);
BOOST_AUTO_TEST_CASE( strided_select_gradient2 ) {
  dynet::ComputationGraph cg;
  for(int from=0;from<2;from++){
    for(int to=from+1;to<4;to++){
      for(int stride=1;stride<4;stride++){
        const vector<int> strides = {stride,stride,stride};
        const vector<int> to_range = {to,to,to};
        const vector<int> from_range = {from,from,from};
        Expression x1 = parameter(cg, param_cube1);
        Expression y = strided_select(x1, strides, from_range, to_range);
        Expression z = to_scalar(y);
        BOOST_CHECK(check_grad(mod, z, 0));
      }
    }
  }
}
// Expression strided_select(const Expression& x, vector<unsigned>& indices);
BOOST_AUTO_TEST_CASE( strided_select_gradient3 ) {
  dynet::ComputationGraph cg;
  for(int from=0;from<2;from++){
    for(int stride=1;stride<4;stride++){
      const vector<int> strides = {stride,stride,stride};
      const vector<int> from_range = {from,from,from};
      Expression x1 = parameter(cg, param_cube1);
      Expression y2 = strided_select(x1, strides, from_range);
      Expression z = to_scalar(y2);
      BOOST_CHECK(check_grad(mod, z, 0));
    }
  }
}
// Expression strided_select(const Expression& x, vector<unsigned>& indices);
BOOST_AUTO_TEST_CASE( strided_select_gradient4 ) {
  dynet::ComputationGraph cg;
  for(int from=0;from<2;from++){
    for(int to=from+1;to<4;to++){
      for(int stride=1;stride<4;stride++){
        const vector<int> strides = {stride,1,stride,stride};
        const vector<int> from_range = {from,0,from,from};
        const vector<int> to_range = {to,1,to,to};
        Expression x1 = reshape(parameter(cg, param_cube1), Dim({3,1,3},3));
        Expression y = strided_select(x1, strides, from_range, to_range);
        Expression z = sum_batches(to_scalar(y));
        BOOST_CHECK(check_grad(mod, z, 0));
      }
    }
  }
}
// Expression strided_select(const Expression& x, vector<unsigned>& indices);
BOOST_AUTO_TEST_CASE( strided_select_gradient5 ) {
  dynet::ComputationGraph cg;
  for(int from=0;from<2;from++){
    for(int to=from+1;to<4;to++){
      for(int stride=1;stride<4;stride++){
        const vector<int> strides = {stride,stride};
        const vector<int> from_range = {from,from};
        const vector<int> to_range = {to,to};
        Expression x1 = reshape(parameter(cg, param_cube1), Dim({3,3,3,1},1));
        Expression y = strided_select(x1, strides, from_range, to_range);
        Expression z = sum_batches(to_scalar(y));
        BOOST_CHECK(check_grad(mod, z, 0));
      }
    }
  }
}

// Expression sum_elems(x);
BOOST_AUTO_TEST_CASE( sum_elems_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression z = sum_elems(x);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression mean_elems(x);
BOOST_AUTO_TEST_CASE( mean_elems_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression z = mean_elems(x);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression moment_elems(x, r);
BOOST_AUTO_TEST_CASE( moment_elems_gradient ) {
  for (unsigned r = 2; r < 5; r++) {
    dynet::ComputationGraph cg;
    Expression x = parameter(cg, param4);
    Expression z = moment_elems(x, r);
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression std_elems(x);
BOOST_AUTO_TEST_CASE( std_elems_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression z = std_elems(x);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sum_batches(x);
BOOST_AUTO_TEST_CASE( sum_batches_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression y = reshape(x, Dim({1}, 6));
  Expression z = sum_batches(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression mean_batches(x);
BOOST_AUTO_TEST_CASE( mean_batches_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression y = reshape(x, Dim({1}, 6));
  Expression z = mean_batches(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression mean_batches(x);
BOOST_AUTO_TEST_CASE( mean_batches_gradient_multidim ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression y = reshape(x, Dim({1, 2}, 3));
  Expression z = mean_batches(y);
  z = mean_dim(z, {1});
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression moment_batches(x, r);
BOOST_AUTO_TEST_CASE( moment_batches_gradient ) {
  for (unsigned r = 2; r < 5; r++) {
    dynet::ComputationGraph cg;
    Expression x = parameter(cg, param4);
    Expression y = reshape(x, Dim({1}, 6));
    Expression z = moment_batches(y, r);
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression sum_dim(x, r);
BOOST_AUTO_TEST_CASE( sum_dim_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_cube1);
  Expression z = x;
  for (unsigned d = 3; d > 0; d--)
    z = sum_dim(z, vector<unsigned>({d - 1}), false);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression std_batches(x);
BOOST_AUTO_TEST_CASE( std_batches_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param4);
  Expression y = reshape(x, Dim({1}, 6));
  Expression z = std_batches(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression mean_dim(x);
BOOST_AUTO_TEST_CASE( mean_dim_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_cube1);
  Expression z = x;
  for (unsigned d = 3; d > 0; d--)
    z = mean_dim(z, {d - 1});
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression moment_dim(x, r);
BOOST_AUTO_TEST_CASE( moment_dim_gradient ) {
  for (unsigned r = 2; r < 5; r++) {
    dynet::ComputationGraph cg;
    Expression x = parameter(cg, param_cube1);
    Expression z = x;
    for (unsigned d = 3; d > 0; d--)
      z = moment_dim(z, vector<unsigned>({d - 1}), r, false);
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression moment_dim(x, r);
BOOST_AUTO_TEST_CASE( moment_dim_gradient2 ) {
  for (unsigned r = 2; r < 5; r++){
    dynet::ComputationGraph cg;
    Expression z = dynet::reshape(parameter(cg, param_cube2), Dim({3,3,3}, 2)) / 10;
    for (unsigned d = 3; d > 0; d--)
      z = moment_dim(z, vector<unsigned>({d - 1}), r, false);
    z = moment_dim(z, vector<unsigned>({}), r, true);
    BOOST_CHECK(check_grad(mod, z, 0));
  }
}

// Expression moment_dim(x, r);
BOOST_AUTO_TEST_CASE( moment_dim_gradient3 ) {
  for (unsigned r=1;r<5;r++){
    dynet::ComputationGraph cg;
    Expression x = dynet::reshape(parameter(cg, param_cube2), Dim({27}, 2))/10;

    Expression y = moment_dim(x, vector<unsigned>({0}), r, true);

    Expression z = moment_dim(x, vector<unsigned>({0}), r, false);
    z = moment_dim(z, vector<unsigned>({}), r, true);

    BOOST_CHECK(check_grad(mod, y, 0));
    BOOST_CHECK(check_grad(mod, z, 0));
    if(r==1) BOOST_CHECK_CLOSE(as_scalar(y.value()), as_scalar(z.value()), 0.001);
  }
}

// Expression moment_dim(x, r);
BOOST_AUTO_TEST_CASE( moment_dim_gradient4 ) {
  for (unsigned r=1;r<5;r++){
    dynet::ComputationGraph cg;
    Expression x = dynet::reshape(parameter(cg, param_cube2), Dim({3,9}, 2)) / 10;

    Expression y = moment_dim(x, vector<unsigned>({0,1}), r, true);

    Expression z = moment_dim(x, vector<unsigned>({0,1}), r, false);
    z = moment_dim(z, vector<unsigned>({}), r, true);

    BOOST_CHECK(check_grad(mod, y, 0));
    BOOST_CHECK(check_grad(mod, z, 0));
    if(r==1) BOOST_CHECK_CLOSE(as_scalar(y.value()), as_scalar(z.value()), 0.001);
  }
}

// Expression std_dim(x);
BOOST_AUTO_TEST_CASE( std_dim_gradient3 ) {
  dynet::ComputationGraph cg;
  Expression x = dynet::reshape(parameter(cg, param_cube2), Dim({27}, 2))/10;

  Expression y = std_dim(x, vector<unsigned>({0}), true);

  Expression z = std_dim(x, vector<unsigned>({0}), false);
  z = std_dim(z, vector<unsigned>({}), true);

  BOOST_CHECK(check_grad(mod, y, 0));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression std_dim(x);
BOOST_AUTO_TEST_CASE( std_dim_gradient4 ) {
  dynet::ComputationGraph cg;
  Expression x = dynet::reshape(parameter(cg, param_cube2), Dim({3,9}, 2))/10;

  Expression y = std_dim(x, vector<unsigned>({0,1}), true);

  Expression z = std_dim(x, vector<unsigned>({0,1}), false);
  z = std_dim(z, vector<unsigned>({}), true);

  BOOST_CHECK(check_grad(mod, y, 0));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression std_dim(x);
BOOST_AUTO_TEST_CASE( std_dim_value ) {
  dynet::ComputationGraph cg;
  Expression x = dynet::reshape(parameter(cg, param_cube1), Dim({3,3}, 3));
  Expression y = std_dim(x, vector<unsigned>({0}), true);
  Expression z = mean_dim(y, vector<unsigned>({0}), false);

  BOOST_CHECK_CLOSE(as_scalar(z.value()), 0.128319368, 0.1);
}

// Expression mean_dim(x);
BOOST_AUTO_TEST_CASE( std_dim_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param_cube1);
  Expression z = x;
  for (unsigned d = 3; d > 0; d--)
    z = std_dim(z, {d - 1});
  BOOST_CHECK(check_grad(mod, z, 0));
}


// Expression layer_norm(x,g,b);
BOOST_AUTO_TEST_CASE( layer_norm_backward_gradient ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param1);
  Expression g = parameter(cg, param2);
  Expression b = parameter(cg, param3);
  Expression y = layer_norm(x, g, b);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression layer_norm(x,g,b);
BOOST_AUTO_TEST_CASE( layer_norm_forward ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, param1);
  Expression g = input(cg, Dim({3}), ones3_vals);
  Expression b = zeroes(cg, Dim({3}));
  Expression y = layer_norm(x, g, b);
  float mu = abs(as_scalar((sum_elems(y) / 3.0).value()));
  float std = as_scalar(sqrt(sum_elems(square(y)) / 3.0).value());
  BOOST_CHECK_LT(mu, 1e-6);
  BOOST_CHECK_CLOSE(std, 1, 0.01);
}

// Expression weight_norm(x,g);
BOOST_AUTO_TEST_CASE( weight_norm_forward ) {
  dynet::ComputationGraph cg;
  Expression w = parameter(cg, param_square1);
  Expression g = parameter(cg, param_scalar1);
  Expression y = weight_norm(w, g);
  float norm = as_scalar(sqrt(sum_elems(square(y))).value());
  BOOST_CHECK_CLOSE(norm, 2.2, 0.01);
}

// Expression layer_norm(x,g);
BOOST_AUTO_TEST_CASE( weight_norm_backward_gradient ) {
  dynet::ComputationGraph cg;
  Expression w = parameter(cg, param_square1);
  Expression g = parameter(cg, param_scalar1);
  Expression y = weight_norm(w, g);
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression sparse_input(vector<unsigned int>& ids, vector<float>& src, float def);
BOOST_AUTO_TEST_CASE( sparse_input_test ) {
  dynet::ComputationGraph cg;
  std::vector<unsigned int> ids = {0, 4};
  Expression z = input(cg, Dim({3}, 2), ids, ones2_vals, 0.5);
  std::vector<float> exp = {1.0f, 0.5f, 0.5f, 0.5f, 1.0f, 0.5f};
  std::vector<float> act = as_vector(cg.forward(z));
  assert(exp.size() == act.size());
  for (size_t i = 0; i < exp.size(); ++i)
    BOOST_CHECK_CLOSE(exp[i], act[i], 0.001);
}

// Expression one_hot(ComputationGraph& g, unsigned int d, unsigned int idx, Device *device = dynet::default_device);
BOOST_AUTO_TEST_CASE( one_hot_test ) {
  dynet::ComputationGraph cg;
  unsigned int idx = 5;
  unsigned int d = 10;
  Expression z = one_hot(cg, d, idx);
  std::vector<float> values = as_vector(cg.forward(z));
  BOOST_CHECK_EQUAL(d, values.size());
  for (size_t i = 0; i < d; ++i)
    BOOST_CHECK_EQUAL(values[i], i == idx ? 1.0 : 0.0);
}
// Expression one_hot(ComputationGraph& g, unsigned int d, unsigned int batch_size, const std::vector<unsigned int>& ids, Device *device = dynet::default_device);
BOOST_AUTO_TEST_CASE( batched_one_hot_test ) {
  dynet::ComputationGraph cg;
  vector<unsigned int> idxs = {1, 6};
  unsigned int d = 10;
  unsigned int batch_size = idxs.size();
  Expression z = one_hot(cg, d, idxs);
  std::vector<float> values = as_vector(cg.forward(z));
  BOOST_CHECK_EQUAL(d * batch_size, values.size());
  for (size_t b = 0; b < batch_size; ++b)
    for (size_t i = 0; i < d; ++i)
      BOOST_CHECK_EQUAL(values[b * d + i], (b * d + i ==  1 || b * d + i == 16 ? 1.0 : 0.0));
}

// Expression lookup();
BOOST_AUTO_TEST_CASE( lookup_test ) {
  dynet::ComputationGraph cg;
  Expression x1 = lookup(cg, lookup1, (unsigned)0);
  Expression x2 = lookup(cg, lookup1, (unsigned)2);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression lookup();
BOOST_AUTO_TEST_CASE( lookup_highdim_batched_test ) {
  dynet::ComputationGraph cg;
  Expression x = lookup(cg, lookup4, {0, 2});
  Expression z = sum_batches(to_scalar(x));
  BOOST_CHECK(check_grad(mod, z, 0));
}

// Expression lookup();
BOOST_AUTO_TEST_CASE( lookup_autobatch_dim_test ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 1;
  dynet::ComputationGraph cg;
  Expression x1 = lookup(cg, lookup1, (unsigned)0);
  Expression x2 = lookup(cg, lookup2, (unsigned)5);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
  dynet::autobatch_flag = autobatch_cache;
}

// Expression lookup();
BOOST_AUTO_TEST_CASE( lookup_autobatch_diffmodel_test ) {
  auto autobatch_cache = dynet::autobatch_flag;
  dynet::autobatch_flag = 1;
  dynet::ComputationGraph cg;
  Expression x1 = lookup(cg, lookup1, (unsigned)0);
  Expression x2 = lookup(cg, lookup3, (unsigned)5);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  BOOST_CHECK(check_grad(mod, z, 0));
  dynet::autobatch_flag = autobatch_cache;
}

// Expression lookup();
BOOST_AUTO_TEST_CASE( lookup_autobatch_and_manbatch_test ) {
  auto autobatch_cache = dynet::autobatch_flag;
  for (dynet::autobatch_flag = 0; dynet::autobatch_flag < 2; ++dynet::autobatch_flag) {
    dynet::ComputationGraph cg;
    Expression x1 = lookup(cg, lookup1, {0, 1});
    Expression x2 = lookup(cg, lookup1, {2, 0});
    Expression y = x1 + x2;
    Expression z = sum_batches(to_scalar(y));
    BOOST_CHECK(check_grad(mod, z, 0));
  }
  dynet::autobatch_flag = autobatch_cache;
}

// Expression parameter() with lookup parameter input;
BOOST_AUTO_TEST_CASE( lookup_matrix_test ) {
  dynet::ComputationGraph cg;
  Expression x = parameter(cg, lookup1);
  Expression z = to_scalar(x);
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_CASE( backward_test ) {
  dynet::ComputationGraph cg;
  Expression x1 = lookup(cg, lookup1, (unsigned)0);
  Expression x2 = lookup(cg, lookup1, (unsigned)2);
  Expression y = x1 + x2;
  Expression z = to_scalar(y);
  cg.backward(z);
}

BOOST_AUTO_TEST_CASE( gradient_value_test ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression l = dot_product(x1, x2);
  cg.backward(l);
  vector<float> x1_g1 = as_vector(x1.gradient());
  vector<float> x1_g2 = as_vector(param1.get_storage().g);

  for (unsigned i = 0; i < 3; i++) {
    BOOST_CHECK_CLOSE(x1_g1[i], x1_g2[i], 0.001);
  }

}

BOOST_AUTO_TEST_CASE( gradient_sanity_test ) {
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression l = dot_product(x1, x2);
  cg.forward(l);
  BOOST_CHECK_THROW(x1.gradient() , std::runtime_error);
}

// This just makes sure that nothing crashes
BOOST_AUTO_TEST_CASE( random_gumbel_test ) {
  dynet::ComputationGraph cg;
  Expression x1 = random_gumbel(cg, {20});
  x1.value();
}

BOOST_AUTO_TEST_CASE( sanity_test ) {
  Expression x;
  {
    dynet::ComputationGraph cg;
    x = input(cg, {3}, ones3_vals);
  }
  BOOST_CHECK_THROW(x.value() , std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
