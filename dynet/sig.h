#ifndef DYNET_SIG_H
#define DYNET_SIG_H

#define DYNET_MAX_SIG 100

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

//
//struct SigString {
//  SigString(int which) : which(which), tail(data) { }
//  SigString() : which(which), tail(data) { }
//  int which;
//  int data[DYNET_MAX_SIG];
//  int* tail;
//
//  void add_node(VariableIndex i) { *(tail++) = -(int)i; }
//  void add_dim(const Dim &d) {
//    *(tail++) = -(int)d.nd;
//    memcpy(tail, d.d, d.nd * sizeof(unsigned int));
//    tail += d.nd; /* * sizeof(unsigned int) / sizeof(int) */
//  }
//};
//
//inline bool operator<(const SigString& a, const SigString& b) {
//  auto c = cmp(a.which, b.which);
//  if(c != 0) return c < 0;
//  ptrdiff_t a_size = a.tail - a.data;
//  c = cmp(a_size, (ptrdiff_t)(b.tail - b.data));
//  if(c != 0) return c < 0;
//  return memcmp(a.data, b.data, a_size * sizeof(int)) < 0;
//}
//
//inline bool operator==(const SigString& a, const SigString& b) {
//  if(a.which != b.which) return false;
//  ptrdiff_t a_size = a.tail - a.data;
//  if(a_size != (ptrdiff_t)(b.tail - b.data)) return false;
//  return memcmp(a.data, b.data, a_size * sizeof(int)) == 0;
//}
//
//inline bool operator!=(const SigString& a, const SigString& b) { return !(a == b); }
//
//struct SigHash {
//  SigHash(int which) : value(0xcc9e2d51 ^ which) { }
//  SigHash() : value(0xcc9e2d51) { }
//  int hash;
//
//  // sbdm hash
//  inline void add_int(int i) {
//    value = i + (hash << 6) + (hash << 16) - hash;
//  }
//  void add_node(VariableIndex i) { add_int((int)i); }
//  void add_dim(const Dim &d) {
//    add_int(-(int)d.nd);
//    for(size_t i = 0; < d.nd; ++i)
//      add_int((int)d.d[i]);
//  }
//
//};
//
//inline bool operator<(const SigHash& a, const SigHash& b) {
//  return a.value < b.value;
//}
//inline bool operator==(const SigHash& a, const SigHash& b) {
//  return a.value == b.value;
//}
//inline bool operator!=(const SigHash& a, const SigHash& b) { return a.value != b.value; }
//*/
//

//
//template <class Sig>
//struct SigLinearMap {
//  SigLinearMap() { sigs.resize(50); }
//
//template <class Sig>
//struct SigTreeMap {
//  SigTreeMap() { }
//  int get_idx(Sig &s) {
//    auto it = sigs.find(s);
//    if(it != sigs.end()) return sigs->second;
//    sigs.insert(make_pair(s, (int)sigs.size()));
//    return sigs.size()-1;
//  }
//  std::map<Sig, int> sigs;
//};
//

// struct Sig {
//   Sig(int which) : which(which), nn(0), nd(0) { }
//   Sig() : which(0), nn(0), nd(0) { }
//   const unsigned short which;
//   unsigned short nn; 
//   unsigned short nd;
//   Dim dims[10];
//   unsigned int node_ids[10];
// 
//   void add_node(VariableIndex i) { node_ids[nn++]=i; }
//   void add_dim(Dim &d)           { dims[nd++]=d;     }
// };
// 

} // namespace dynet

#endif
