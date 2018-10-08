#ifndef DYNET_C_TEST_UTILS_H_
#define DYNET_C_TEST_UTILS_H_

#include <dynet_c/init.h>
#include <dynet_c/status.h>

#include <iostream>
#include <vector>

namespace test_utils {

void init_dynet() {
  const char *argv[] = {"DynetCTest", "--dynet-mem", "512"};
  ::dynetDynetParams_t *params;
  ::dynetExtractDynetParams(3, const_cast<char**>(argv), false, &params);
  ::dynetInitialize(params);
  ::dynetDeleteDynetParams(params);
}

void show_message() {
  std::size_t length = 0u;
  ::dynetGetMessage(nullptr, &length);
  char str[length];
  ::dynetGetMessage(str, &length);
  std::cout << str << std::endl;
  ::dynetResetStatus();
}

}  // namespace test_utils

#endif  // DYNET_C_TEST_UTILS_H_
