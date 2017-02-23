#ifndef DYNET_TEST_H_
#define DYNET_TEST_H_

#include <vector>
#include <unordered_map>
#include <boost/test/unit_test.hpp>

#define TOL 1e-7
#define DYNET_CHECK_EQUAL(a, b) equal_check(a, b)
#define DYNET_CHECK_CLOSE(a, b) close_check(a, b)

template <class T, class U>
void equal_check(T a, U b) {
  BOOST_CHECK_EQUAL(a, b);
}

template <class T>
void equal_check(std::vector<T> a,
                 std::vector<T> b) {
  BOOST_CHECK_EQUAL(a.size(), b.size());
  for(size_t i = 0; i < a.size(); ++i) {
    BOOST_CHECK_EQUAL(a[i], b[i]);
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

#endif
