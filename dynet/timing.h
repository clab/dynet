#ifndef _TIMING_H_
#define _TIMING_H_

#include <iostream>
#include <string>
#include <chrono>

namespace dynet {

struct Timer {
  Timer(const std::string& msg) : msg(msg), start(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    auto stop = std::chrono::high_resolution_clock::now();
    std::cerr << '[' << msg << ' ' << std::chrono::duration<double, std::milli>(stop-start).count() << " ms]\n";
  }
  std::string msg;
  std::chrono::high_resolution_clock::time_point start;
};

struct Timing {
  Timing() : _start(std::chrono::high_resolution_clock::now()) {}
  ~Timing() { }
  double stop() {
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(stop-_start).count();
  }
  void start() { _start = std::chrono::high_resolution_clock::now(); }
  std::chrono::high_resolution_clock::time_point _start;
};

} // namespace dynet

#endif
