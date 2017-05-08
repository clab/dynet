#ifndef DYNET_SIG_H
#define DYNET_SIG_H

namespace dynet {
struct Sig {
  Sig(short which) : which(which), nn(0), nd(0) { }
  Sig() : which(0), nn(0), nd(0) { }
  const unsigned short which;
  unsigned short nn; 
  unsigned short nd;
  Dim dims[10];
  unsigned int node_ids[10];

  void add_node(VariableIndex i) { node_ids[nn++]=i; }
  void add_dim(Dim &d)           { dims[nd++]=d;     }
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

} // namespace dynet

#endif
