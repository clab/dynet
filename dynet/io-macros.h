#ifndef DYNET_IO_MACROS__
#define DYNET_IO_MACROS__

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#define DYNET_SERIALIZE_IMPL(MyClass) \
  template void MyClass::serialize<boost::archive::text_oarchive>(boost::archive::text_oarchive &ar, const unsigned int); \
  template void MyClass::serialize<boost::archive::text_iarchive>(boost::archive::text_iarchive &ar, const unsigned int); \
  template void MyClass::serialize<boost::archive::binary_oarchive>(boost::archive::binary_oarchive &ar, const unsigned int); \
  template void MyClass::serialize<boost::archive::binary_iarchive>(boost::archive::binary_iarchive &ar, const unsigned int);

#define DYNET_SAVELOAD_IMPL(MyClass) \
  template void MyClass::save<boost::archive::text_oarchive>(boost::archive::text_oarchive &ar, const unsigned int) const; \
  template void MyClass::load<boost::archive::text_iarchive>(boost::archive::text_iarchive &ar, const unsigned int); \
  template void MyClass::save<boost::archive::binary_oarchive>(boost::archive::binary_oarchive &ar, const unsigned int) const; \
  template void MyClass::load<boost::archive::binary_iarchive>(boost::archive::binary_iarchive &ar, const unsigned int);

#endif
