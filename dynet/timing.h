#ifndef _TIMING_H_
#define _TIMING_H_

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <map>
#include <algorithm>
#include <iterator>

// for sorting timer output
template<typename A, typename B> std::pair<B,A> flip_pair(const std::pair<A,B> &p) { return std::pair<B,A>(p.second, p.first); }
template<typename A, typename B> std::multimap<B,A> flip_map(const std::map<A,B> &src) { std::multimap<B,A> dst; std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),flip_pair<A,B>);return dst;}

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
      std::cout << "Timing Info:" << std::endl;
      show();
    }
  }
  void start(std::string name) {
    Timing t;
    timers[name] = t;
  }
  void stop(std::string name) {
    cumtimes[name] += (timers[name]).stop();
  }
  void show() {
    std::multimap<double, std::string> cumtimes_dst = flip_map(cumtimes);
    double total_time = 0.0;
    for (auto &item : cumtimes_dst) {
      total_time += item.first;
    }
    for (auto &item : cumtimes_dst) {
      std::cout << std::setprecision(4) << std::setw(11) << item.first << '\t' << (100.0*item.first/total_time) << "%\t" << item.second << std::endl;
    }
    std::cout << std::setprecision(4) << std::setw(11) << total_time << "\t100%\t(total time)" << std::endl;
  }
  std::map<std::string, double> cumtimes;
  std::map<std::string, Timing> timers;
};

} // namespace dynet

#endif
