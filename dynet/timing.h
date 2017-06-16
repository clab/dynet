#ifndef _TIMING_H_
#define _TIMING_H_

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <map>

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

class NamedTimer {
public:
  ~NamedTimer() { 
    if (timers.size()>0) {
      std::cout << "Timing Info:" << std::endl; show();
    }
  }
  void start(std::string name) { Timing t; timers[name] = t; }
  void stop(std::string name) { cumtimes[name] += (timers[name]).stop(); }
  void show() { for (auto &item : cumtimes) { std::cout << std::setprecision(4) << std::setw(11) << item.second << '\t' << item.first << std::endl; } }
  std::map<std::string, double> cumtimes;
  std::map<std::string, Timing> timers;
};

} // namespace dynet

#endif
