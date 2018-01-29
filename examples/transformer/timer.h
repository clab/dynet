#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <map>

struct MyTimer {
  MyTimer(const std::string& msg) : msg(msg), start(std::chrono::high_resolution_clock::now()) {}
  ~MyTimer() {
  }
  void show(){
    auto stop = std::chrono::high_resolution_clock::now();
    std::cerr << '[' << msg << ' ' << std::chrono::duration<double, std::milli>(stop-start).count() << " ms]\n";
  }
  void reset(){
    start = std::chrono::high_resolution_clock::now();
  }	
  double elapsed() {
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(stop-start).count();// in ms
  }
  std::string msg;
  std::chrono::high_resolution_clock::time_point start;
};


