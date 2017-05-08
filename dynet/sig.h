#ifndef DYNET_SIG_H
#define DYNET_SIG_H


namespace dynet {

  namespace nt {
    enum NodeType { 
      tanh=1, sqrt, abs, erf, square, cube, exp, loggamma, log, nobackprop, flipgradient, identity, negate, rectify, logistic, softsign,
      plus_const, concat, matmul, cmult, affine, sum, squared_distance, pnls, pickrange,
      input, scalar_input, lookup
    };
  }

struct Sig {
  Sig(short which) : which(which), nn(0), nd(0) { }
  Sig() : which(0), nn(0), nd(0) { }
  const unsigned short which;
  unsigned short nn; 
  unsigned short nd;
  Dim dims[10];
  unsigned node_ids[10];

  void add_node(unsigned i) { node_ids[nn++]=i; }
  // TODO add_dim is NOT SAME as dim.print_profile(oss)
  void add_dim(const Dim &d)           { dims[nd++]=d;     }
};

inline bool operator==(const Sig& a, const Sig& b) {
  if (a.which == b.which && a.nn == b.nn && a.nd == b.nd) {
    for (int i = 0; i < a.nn; ++i) { if (a.node_ids[i] != b.node_ids[i]) return false; }
    for (int i = 0; i < a.nd; ++i) { if (a.dims[i] != b.dims[i]) return false; }
    return true;
  } else {
    return false;
  }
}
inline bool operator!=(const Sig& a, const Sig& b) { return !(a == b); }

struct SigMap {
  SigMap() { sigs.resize(50); Sig s; sigs.push_back(s); }
  int get_idx(Sig &s) {
    for (int i=0; i<sigs.size(); ++i) {
      if (sigs[i]==s) return i;
    }
    sigs.push_back(s);
    return sigs.size()-1;
  }
  int size() { return sigs.size(); }
  std::vector<Sig> sigs;
};

} // namespace dynet

#endif
