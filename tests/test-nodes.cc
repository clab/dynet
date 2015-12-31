#include <cnn/cnn.h>
#include <cnn/expr.h>
#include <cnn/grad-check.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace cnn;
using namespace cnn::expr;
using namespace std;

struct NodeTest {
  NodeTest() {
    // set up some dummy arguments to cnn
    for (auto x : {"NodeTest", "--cnn-mem", "10"}) {
      av.push_back(strdup(x));
    }
    char **argv = &av[0];
    int argc = av.size();
    cnn::Initialize(argc, argv);
    ones3_vals = {1.f,1.f,1.f};
    ones2_vals = {1.f,1.f};
    // Create parameters
    std::vector<float> param1_vals = {1.1f,-2.2f,3.3f};
    std::vector<float> param2_vals = {2.2f,3.4f,-1.2f};
    std::vector<float> param_scalar1_vals = {2.2f};
    std::vector<float> param_scalar2_vals = {1.1f};
    param1 = mod.add_parameters({3});
    TensorTools::SetElements(param1->values,param1_vals);
    param2 = mod.add_parameters({3});
    TensorTools::SetElements(param2->values,param2_vals);
    param_scalar1 = mod.add_parameters({1});
    TensorTools::SetElements(param_scalar1->values,param_scalar1_vals);
    param_scalar2 = mod.add_parameters({1});
    TensorTools::SetElements(param_scalar2->values,param_scalar2_vals);
  }
  ~NodeTest() {
    for (auto x : av) free(x);
  }
  std::vector<float> ones3_vals, ones2_vals;
  std::vector<char*> av;
  cnn::Model mod;
  cnn::Parameters *param1, *param2, *param_scalar1, *param_scalar2;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(node_test, NodeTest);


// Expression operator-(const Expression& x);
BOOST_AUTO_TEST_CASE( negate_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = -x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator+(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( add_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1+x2;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator+(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( addscalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1+2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator+(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalaradd_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0+x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator-(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( subtract_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1+x2;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator-(real x, const Expression& y);
BOOST_AUTO_TEST_CASE( scalarsubtract_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0-x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator-(const Expression& x, real y);
BOOST_AUTO_TEST_CASE( subtractscalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1-2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( multiply_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = x1*transpose(x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  ones3 * y * transpose(ones3);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression operator*(const Expression& x, float y);
BOOST_AUTO_TEST_CASE( multiplyscalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1*2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// inline Expression operator*(float y, const Expression& x) { return x * y; }
BOOST_AUTO_TEST_CASE( scalarmultiply_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = 2.0*x1;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// inline Expression operator/(const Expression& x, float y) { return x * (1.f / y); }
BOOST_AUTO_TEST_CASE( dividescalar_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = x1/2.0;
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression cdiv(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cdiv_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cdiv(x1, x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression colwise_add(const Expression& x, const Expression& bias);
BOOST_AUTO_TEST_CASE( colwise_add_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = colwise_add(x1 * transpose(x2), x2);
  Expression ones3 = input(cg, {1,3}, ones3_vals);
  ones3 * y * transpose(ones3);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression contract3d_1d(const Expression& x, const Expression& y);
// TODO

// Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b);
// TODO

// Expression sqrt(const Expression& x);
BOOST_AUTO_TEST_CASE( sqrt_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = sqrt(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression erf(const Expression& x);
BOOST_AUTO_TEST_CASE( erf_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = erf(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression tanh(const Expression& x);
BOOST_AUTO_TEST_CASE( tanh_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = tanh(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression exp(const Expression& x);
BOOST_AUTO_TEST_CASE( exp_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = exp(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression square(const Expression& x);
BOOST_AUTO_TEST_CASE( square_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = square(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression cube(const Expression& x);
BOOST_AUTO_TEST_CASE( cube_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = cube(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression lgamma(const Expression& x);
BOOST_AUTO_TEST_CASE( lgamma_gradient ) {
  cnn::ComputationGraph cg;
  Expression x2 = parameter(cg, param2);
  Expression y = lgamma(x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression log(const Expression& x);
BOOST_AUTO_TEST_CASE( log_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression logistic(const Expression& x);
BOOST_AUTO_TEST_CASE( logistic_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = logistic(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression rectify(const Expression& x);
BOOST_AUTO_TEST_CASE( rectify_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = rectify(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression hinge(const Expression& x, unsigned index, float m = 1.0);
BOOST_AUTO_TEST_CASE( hinge_gradient ) {
  unsigned index = 0;
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  hinge(x1, index, 0.5);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression hinge(const Expression& x, const unsigned* pindex, float m = 1.0);
BOOST_AUTO_TEST_CASE( hingeptr_gradient ) {
  unsigned index = 0;
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  hinge(x1, &index, 0.5);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression log_softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( log_softmax_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log_softmax(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression log_softmax(const Expression& x, const std::vector<unsigned>& restriction);
BOOST_AUTO_TEST_CASE( restricted_log_softmax_gradient ) {
  vector<unsigned> restriction = {1,2};
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = log_softmax(x1, restriction);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression softmax(const Expression& x);
BOOST_AUTO_TEST_CASE( softmax_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = softmax(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression softsign(const Expression& x);
BOOST_AUTO_TEST_CASE( softsign_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = softsign(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression pow(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( pow_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x_scalar1 = parameter(cg, param_scalar1);
  Expression y = pow(x1, x_scalar1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression min(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( min_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = min(x1, x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression max(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( max_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = max(x1, x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression noise(const Expression& x, real stddev);
BOOST_AUTO_TEST_CASE( noise_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = noise(x1, 0.5);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression dropout(const Expression& x, real p);
BOOST_AUTO_TEST_CASE( dropout_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = dropout(x1, 0.5);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression block_dropout(const Expression& x, real p);
// TODO

// Expression reshape(const Expression& x, const Dim& d);
BOOST_AUTO_TEST_CASE( reshape_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = reshape(x1, {1,3});
  y * input(cg, {3}, ones3_vals);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression transpose(const Expression& x);
BOOST_AUTO_TEST_CASE( transpose_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = softsign(x1);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression trace_of_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( trace_of_product_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  trace_of_product(x1, x2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression cwise_multiply(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( cwise_multiply_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  Expression y = cwise_multiply(x1, x2);
  input(cg, {1,3}, ones3_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression dot_product(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( dot_product_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  dot_product(x1, x2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression squared_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( squared_distance_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  squared_distance(x1, x2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression huber_distance(const Expression& x, const Expression& y, float c = 1.345f);
BOOST_AUTO_TEST_CASE( huber_distance_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  huber_distance(x1, x2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression l1_distance(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( l1_distance_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  l1_distance(x1, x2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression binary_log_loss(const Expression& x, const Expression& y);
BOOST_AUTO_TEST_CASE( binary_log_loss_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param2);
  binary_log_loss(x1, x2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m=1.0);
BOOST_AUTO_TEST_CASE( pairwise_rank_loss_gradient ) {
  cnn::ComputationGraph cg;
  Expression x_scalar1 = parameter(cg, param_scalar1);
  Expression x_scalar2 = parameter(cg, param_scalar2);
  pairwise_rank_loss(x_scalar1, x_scalar2);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// TODO: These are all unimplemented
// Expression poisson_loss(const Expression& x, unsigned y);
// Expression poisson_loss(const Expression& x, const unsigned* py);
// 
// Expression conv1d_narrow(const Expression& x, const Expression& f);
// Expression conv1d_wide(const Expression& x, const Expression& f);
// Expression kmax_pooling(const Expression& x, unsigned k);
// Expression fold_rows(const Expression& x, unsigned nrows=2);
// Expression sum_cols(const Expression& x);
// Expression kmh_ngram(const Expression& x, unsigned n);
// 
// Expression sum_batches(const Expression& x);

// Expression pick(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pick_gradient ) {
  unsigned idx = 1;
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  pick(x1, idx);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression pick(const Expression& x, unsigned* pv);
BOOST_AUTO_TEST_CASE( pickptr_gradient ) {
  unsigned idx = 1;
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  pick(x1, &idx);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression pickrange(const Expression& x, unsigned v, unsigned u);
BOOST_AUTO_TEST_CASE( pickrange_gradient ) {
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression y = pickrange(x1, 0, 2);
  input(cg, {1,2}, ones2_vals) * y;
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression pickneglogsoftmax(const Expression& x, unsigned v);
BOOST_AUTO_TEST_CASE( pickneglogsoftmax_gradient ) {
  unsigned idx = 1;
  cnn::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  pickneglogsoftmax(x1, idx);
  BOOST_CHECK(CheckGrad(mod, cg, 0));
}

// Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> & v);
// TODO: Not covered yet

BOOST_AUTO_TEST_SUITE_END()
