#define BOOST_TEST_MODULE TEST_SERIALIZE

#include <vector>
#include <string>
#include <fstream>
#include <exception>

#include <boost/test/unit_test.hpp>
#include <dynet/io-macros.h>
#include "test.h"

#define EMPTY_FUNC() 

class modelTestA {
 public:
  modelTestA(int _i, double _d,
            std::vector<std::string> _v) : i(_i), d(_d), V(_v) {}
  modelTestA() {}
 public:
  int i;
  double d;
  std::vector<std::string> V;
  DYNET_SERIALIZE_DECLARE()
}; // class modelTestA
DYNET_SERIALIZE_COMMIT(modelTestA, DYNET_SERIALIZE_DEFINE(i, d, V))

class modelTestB : public modelTestA {
 public:
  modelTestB(int _i, double _d,
             std::vector<std::string> _v,
             std::string _s) :
      modelTestA(_i, _d, _v), s(_s) {}
  modelTestB() {}
 public:
  std::string s;
  DYNET_SERIALIZE_DECLARE()
}; // class modelTestB
DYNET_SERIALIZE_COMMIT(modelTestB, DYNET_SERIALIZE_DERIVED_DEFINE(modelTestA, s))

class modelTestC {
 public:
  modelTestC(int _i, double _d,
            std::vector<std::string> _v) : i(_i), d(_d), V(_v) {}
  modelTestC() {}
 public:
  int i;
  double d;
  std::vector<std::string> V;
  DYNET_SERIALIZE_SPLIT_DECLARE()
}; // class modelTestC
DYNET_SERIALIZE_SAVE_COMMIT(modelTestC, DYNET_SERIALIZE_DEFINE(i, d, V))
DYNET_SERIALIZE_LOAD_COMMIT(modelTestC, EMPTY_FUNC(), DYNET_SERIALIZE_DEFINE(i, d, V))

class modelTestD {
 public:
  modelTestD(int _i, double _d,
            std::vector<std::string> _v) : i(_i), d(_d), V(_v) {}
  modelTestD() {}
 public:
  int i;
  double d;
  std::vector<std::string> V; // here, V is a new member variable in version 1
  DYNET_SERIALIZE_DECLARE()
}; // class modelTestD
DYNET_SERIALIZE_COMMIT(modelTestD,
                       DYNET_VERSION_SERIALIZE_DEFINE(0, 1, i, d),
                       DYNET_VERSION_SERIALIZE_DEFINE(1, 2, i, d, V))
DYNET_VERSION_DEFINE(modelTestD, 1)

class modelTestE {
 public:
  modelTestE(int _i, double _d,
            std::vector<std::string> _v) : i(_i), d(_d), V(_v) {}
  modelTestE() {}
 public:
  int i;
  double d;
  std::vector<std::string> V;
}; // class modelTestE
DYNET_NINTRUSIVE_SERIALIZE_DEFINE(modelTestE & obj, obj.i, obj.d, obj.V)

BOOST_AUTO_TEST_CASE( io_macros_serialize ) {
  {
    std::string filename = "/tmp/modelTestA.dump";
    std::ofstream ofs(filename);
    modelTestA obj(1, 3.1415926, {"a", "b", "c"});
    {
      boost::archive::text_oarchive oa(ofs);
      oa << obj;
    }
    modelTestA new_obj;
    {
      std::ifstream ifs(filename);
      boost::archive::text_iarchive ia(ifs);
      ia >> new_obj;
    }
    DYNET_CHECK_EQUAL(new_obj.i, 1);
    DYNET_CHECK_EQUAL(new_obj.d, 3.1415926);
    std::vector<std::string> s = {"a", "b", "c"};
    DYNET_CHECK_EQUAL(new_obj.V, s);
  }
  {
    std::string filename = "/tmp/modelTestB.dump";
    std::ofstream ofs(filename);
    modelTestB obj(1, 3.1415926, {"a", "b", "c"}, "dynet");
    {
      boost::archive::text_oarchive oa(ofs);
      oa << obj;
    }
    modelTestB new_obj;
    {  
      std::ifstream ifs(filename);
      boost::archive::text_iarchive ia(ifs);
      ia >> new_obj;
    }  
    DYNET_CHECK_EQUAL(new_obj.i, 1);
    DYNET_CHECK_EQUAL(new_obj.d, 3.1415926);
    std::vector<std::string> s = {"a", "b", "c"};
    DYNET_CHECK_EQUAL(new_obj.V, s);
    DYNET_CHECK_EQUAL(new_obj.s, "dynet");
  }
}

BOOST_AUTO_TEST_CASE( io_macros_serialize_split ) {
  std::string filename = "/tmp/modelTestC.dump";
  std::ofstream ofs(filename);
  modelTestC obj(1, 3.1415926, {"a", "b", "c"});
  {
    boost::archive::text_oarchive oa(ofs);
    oa << obj;
  }
  modelTestC new_obj;
  {
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> new_obj;
  }
  DYNET_CHECK_EQUAL(new_obj.i, 1);
  DYNET_CHECK_EQUAL(new_obj.d, 3.1415926);
  std::vector<std::string> s = {"a", "b", "c"};
  DYNET_CHECK_EQUAL(new_obj.V, s);
}

BOOST_AUTO_TEST_CASE( io_macros_serialize_version ) {
  std::string filename = "/tmp/modelTestD.dump";
  std::ofstream ofs(filename);
  modelTestD obj(1, 3.1415926, {"a", "b", "c"});
  {
    boost::archive::text_oarchive oa(ofs);
    oa << obj;
  }
  modelTestD new_obj;
  {  
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> new_obj;
  }
  DYNET_CHECK_EQUAL(new_obj.i, 1);
  DYNET_CHECK_EQUAL(new_obj.d, 3.1415926);
  std::vector<std::string> s = {"a", "b", "c"};
  DYNET_CHECK_EQUAL(new_obj.V, s);
}

BOOST_AUTO_TEST_CASE( io_macros_serialize_non_intrusive ) {
  std::string filename = "/tmp/modelTestE.dump";
  std::ofstream ofs(filename);
  modelTestE obj(1, 3.1415926, {"a", "b", "c"});
  {
    boost::archive::text_oarchive oa(ofs);
    oa << obj;
  }
  modelTestE new_obj;
  {  
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> new_obj;
  }
  DYNET_CHECK_EQUAL(new_obj.i, 1);
  DYNET_CHECK_EQUAL(new_obj.d, 3.1415926);
  std::vector<std::string> s = {"a", "b", "c"};
  DYNET_CHECK_EQUAL(new_obj.V, s);
}
