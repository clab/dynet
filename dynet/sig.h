#ifndef DYNET_SIG_H
#define DYNET_SIG_H

#define DYNET_MAX_SIG 100

namespace dynet {

struct Sig {
  Sig(int which) : tail(data+1) { data[0] = which; }
  Sig() : tail(data+1) { data[0] = 0; }
  int data[DYNET_MAX_SIG];
  int* tail;

  void add_node(VariableIndex i) { *(tail++) = -(int)i; }
  void add_dim(const Dim &d) {
    *(tail++) = -(int)d.nd;
    memcpy(tail, d.d, d.nd * sizeof(unsigned int));
    tail += d.nd; /* * sizeof(unsigned int) / sizeof(int) */
  }

};

inline bool operator==(const Sig& a, const Sig& b) {
  ptrdiff_t a_size = a.tail - a.data;
  if(a_size != (ptrdiff_t)(b.tail - b.data)) return false;
  return memcmp(a.data, b.data, a_size * sizeof(int)) == 0;
}

inline bool operator!=(const Sig& a, const Sig& b) { return !(a == b); }

struct SigMap {
  SigMap() { sigs.resize(50); }
  int get_idx(Sig &s) {
    for (int i=0; i<sigs.size(); ++i) {
      if (sigs[i]==s) return i;
    }
    sigs.push_back(s);
    return sigs.size()-1;
  }
  std::vector<Sig> sigs;
};

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
// inline bool operator==(const Sig& a, const Sig& b) {
//   if (a.which == b.which && a.nn == b.nn && a.nd == b.nd) {
//     for (int i = 0; i < a.nn; ++i) { if (a.node_ids[i] != b.node_ids[i]) return false; }
//     for (int i = 0; i < a.nd; ++i) { if (a.dims[i] != b.dims[i]) return false; }
//     return true;
//   } else {
//     return false;
//   }
// }

} // namespace dynet

#endif
