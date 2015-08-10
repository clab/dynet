#pragma once

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/rnnem.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

class Model;

#define RNNEM_MEM_SIZE 32
#define RNNEM_ALIGN_DIM 128

struct NMNBuilder : public RNNBuilder{

  NMNBuilder() = default;
  explicit NMNBuilder(long layers,
      long input_dim,
      long hidden_dim,
      Model* model);

  void rewind_one_step() {
      h.pop_back();
      c.pop_back();
      w.pop_back();
      M.pop_back();
  }
  Expression back() const { return h.back().back(); }
  std::vector<Expression> final_s() const {
      std::vector<Expression> ret = (w.size() == 0 ? w0 : w.back());
      for (auto my_M : final_M()) ret.push_back(my_M);
      for (auto my_c : final_c()) ret.push_back(my_c);
      for (auto my_h : final_h()) ret.push_back(my_h);
      return ret;
  }
  std::vector<Expression> final_c() const { return (c.size() == 0 ? c0 : c.back()); }
  std::vector<Expression> final_w() const { return (w.size() == 0 ? w0 : w.back()); }
  std::vector<Expression> final_M() const { return (M.size() == 0 ? M0 : M.back()); }
  std::vector<Expression> final_h() const {
      return (h.size() == 0 ? h0 : h.back());
  }
 private:
     std::vector<Expression> read_memory(const int& t, const Expression & x_t, const size_t layer);
     vector<Expression> update_memory(const int& t, const Expression & x_t, const size_t layer,
         vector<Expression>& Mt);
     Expression compute_importance_factor(const Expression & r_t, /// retrieved memory content 
         const Expression & e_t, /// erase vector
         const Expression & o_t, /// new content vector
         const size_t layer);

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;

 public:
  // first index is layer, then ...
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<Expression>> h, c, w;
  // for external memeory
  std::vector<std::vector<Expression>> M;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> w0, h0, c0, M0;
  long layers;
  long m_mem_size;
  long m_mem_dim;
  long m_align_dim;
};

} // namespace cnn

