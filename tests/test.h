#ifndef DYNET_TEST_H_
#define DYNET_TEST_H_

#include <vector>
#include <unordered_map>
#include <boost/test/unit_test.hpp>
#include <dynet/model.h>

#define TOL 1e-3
#define DYNET_CHECK_EQUAL(a, b) equal_check(a, b)
#define DYNET_CHECK_CLOSE(a, b) close_check(a, b)

using namespace dynet;

template <class T>
void close_check(T a, T b) {
  BOOST_CHECK_CLOSE(a, b, TOL);
}

template <class V>
void close_check(std::vector<V> a,
                 std::vector<V> b) {
  BOOST_CHECK_EQUAL(a.size(), b.size());
  for(size_t i = 0; i < a.size(); ++i) {
    BOOST_CHECK_CLOSE(a[i], b[i], TOL);
  }
}

template <class K, class V>
void close_check(std::unordered_map<K, V> a,
                 std::unordered_map<K, V> b) {
  BOOST_CHECK_EQUAL(a.size(), b.size());
  for(auto & kv : a) {
    BOOST_CHECK(b.count(kv.first));
    BOOST_CHECK_CLOSE(kv.second, b[kv.first], TOL);
  }
}

void equal_check(Tensor & v1, Tensor & v2) {
  std::vector<real> lv1 = dynet::as_vector(v1);
  std::vector<real> lv2 = dynet::as_vector(v2);
  BOOST_CHECK_EQUAL(lv1.size(), lv2.size());
  for (size_t i = 0; i < lv1.size(); ++i) {
    DYNET_CHECK_CLOSE(lv1[i], lv2[i]);
  }
}

template <class T>
void equal_check(std::vector<T> & a,
                 std::vector<T> & b) {
  BOOST_CHECK_EQUAL(a.size(), b.size());
  for(size_t i = 0; i < a.size(); ++i) {
    BOOST_CHECK_EQUAL(a[i], b[i]);
  }
}

void equal_check(ParameterStorage *p1,
                 ParameterStorage *p2) {
  // BOOST_CHECK_EQUAL(p1->name, p2->name);
  BOOST_CHECK_EQUAL(p1->dim, p2->dim);
  equal_check(p1->values, p2->values);
  equal_check(p1->g, p2->g);
}

void equal_check(LookupParameterStorage *p1,
                 LookupParameterStorage *p2) {
  // BOOST_CHECK_EQUAL(p1->name, p2->name);
  BOOST_CHECK_EQUAL(p1->all_dim, p2->all_dim);
  BOOST_CHECK_EQUAL(p1->dim, p2->dim);
  equal_check(p1->all_values, p2->all_values);
  equal_check(p1->all_grads, p2->all_grads);
}

void equal_check(Parameter & p1,
                 Parameter & p2) {
  equal_check(&p1.get_storage(), &p2.get_storage());
}

void equal_check(LookupParameter & p1,
                 LookupParameter & p2) {
  equal_check(&p1.get_storage(), &p2.get_storage());
}

template <class T, class U>
void equal_check(T a, U b) {
  BOOST_CHECK_EQUAL(a, b);
}

void equal_check(ParameterCollection & model1,
                 ParameterCollection & model2) {
  auto params1 = model1.get_parameter_storages();
  auto params2 = model2.get_parameter_storages();
  BOOST_CHECK_EQUAL(params1.size(), params2.size());
  for (size_t i = 0; i < params1.size(); ++i) {
    equal_check(params1[i], params2[i]);
  }
  auto lookup_params1 = model1.get_lookup_parameter_storages();
  auto lookup_params2 = model2.get_lookup_parameter_storages();
  BOOST_CHECK_EQUAL(lookup_params1.size(), lookup_params2.size());
  for (size_t i = 0; i < lookup_params1.size(); ++i) {
    equal_check(lookup_params1[i], lookup_params2[i]);
  }
}

template <class K, class V>
void equal_check(std::unordered_map<K, V> a,
                 std::unordered_map<K, V> b) {
  BOOST_CHECK_EQUAL(a.size(), b.size());
  for(auto & kv : a) {
    BOOST_CHECK(b.count(kv.first));
    BOOST_CHECK_EQUAL(kv.second, b[kv.first]);
  }
}

#endif
