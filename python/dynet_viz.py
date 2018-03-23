from __future__ import print_function
import sys
import re
from collections import defaultdict

if sys.version_info.major > 2:
  # alias dict.items() as dict.iteritems() in python 3+
  class compat_dict(defaultdict):
    pass

  compat_dict.iteritems = defaultdict.items
  defaultdict = compat_dict
  
  # add xrange to python 3+
  xrange = range

graphviz_items = []

vindex_count = -1
def new_index():
  global vindex_count
  vindex_count += 1
  return vindex_count

def init(random_seed=None): pass

class SimpleConcreteDim(object):
  def __init__(self, nrows, ncols, inferred):
    self.nrows = nrows
    self.ncols = ncols
    self.inferred = inferred
  def __getitem__(self, key): return [self.nrows, self.ncols][key]
  def __iter__(self): return iter([self.nrows, self.ncols])
  def __str__(self): return 'Dim(%s,%s)' % (self.nrows, self.ncols)
  def __eq__(self, other): return isinstance(other, SimpleConcreteDim) and self.nrows==other.nrows and self.ncols==other.ncols
  def __ne__(self, other): return not self==other
  def __hash__(self): return hash((self.nrows, self.ncols))
  def isvalid(self): return True
  def invalid(self): return False

class InvalidConcreteDim(object):
  def __init__(self, a_dim=None, b_dim=None): 
    self.a_dim = a_dim
    self.b_dim = b_dim
  def __getitem__(self, key): return None
  def __repr__(self):
    if self.a_dim is None and self.b_dim is None:
      return 'InvalidDim'
    else:
      return 'InvalidDim(%s, %s)' % (self.a_dim, self.b_dim)
  def __str__(self): return repr(self)
  def isvalid(self): return False
  def invalid(self): return True
  
InvalidDim = InvalidConcreteDim()
  
def make_dim(a, b=None, inferred=False):
  if isinstance(a, InvalidConcreteDim):
    assert b is None
    return a
  elif isinstance(a, SimpleConcreteDim):
    assert b is None
    return SimpleConcreteDim(a.nrows, a.ncols, inferred)
  elif isinstance(a, tuple):
    assert b is None
    assert len(a) == 2, str(a)
    (nrows, ncols) = a
    return SimpleConcreteDim(nrows, ncols, inferred)
  elif b is None:
    assert isinstance(a, int) or (isinstance(a, float) and int(a) == a)
    return SimpleConcreteDim(a, 1, inferred)
  else:
    assert isinstance(a, int) or (isinstance(a, float) and int(a) == a)
    assert isinstance(b, int) or (isinstance(b, float) and int(b) == b)
    return SimpleConcreteDim(a, b, inferred)
  


def ensure_freshness(a):
    if a.cg_version != _cg.version(): raise ValueError("Attempt to use a stale expression.")


def copy_dim(a):
  if a.dim.isvalid():
    return make_dim(a.dim, inferred=True)
  else:
    return InvalidDim
def ensure_same_dim(a,b):
  if a.dim.invalid() or b.dim.invalid():
    return InvalidDim
  elif a.dim==b.dim:
    return copy_dim(a)
  else:
    return InvalidConcreteDim(a.dim,b.dim)
def ensure_mul_dim(a,b):
  if a.dim.invalid() or b.dim.invalid():
    return InvalidDim
  elif a.dim[1]==b.dim[0]:
    return make_dim(a.dim[0], b.dim[1], inferred=True)
  else:
    return InvalidConcreteDim(a.dim,b.dim)
def ensure_all_same_dim(xs):
  for x in xs:
    if x.dim.invalid():
      return InvalidDim
  dim0 = xs[0].dim
  for x in xs[1:]:
    if dim0 != x.dim:
      return InvalidConcreteDim(dim0, x.dim)
  return copy_dim(xs[0])
    

def _add(a, b):       return GVExpr('add', [a,b], ensure_same_dim(a,b))
def _mul(a, b):       return GVExpr('mul', [a,b], ensure_mul_dim(a,b))
def _neg(a):          return GVExpr('neg',       [a],   copy_dim(a))
def _scalarsub(a, b): return GVExpr('scalarsub', [a,b], copy_dim(b))
def _cadd(a, b):      return GVExpr('cadd', [a,b], copy_dim(a))
def _cmul(a, b):      return GVExpr('cmul', [a,b], copy_dim(a))
def _cdiv(a, b):      return GVExpr('cdiv', [a,b], copy_dim(a))

class Expression(object): #{{{
  def __init__(self, name, args, dim):
    self.name = name
    self.args = args
    self.dim = dim
    self.vindex = new_index()
    self.cg_version = cg().version()

  def cg(self): return cg()
  def get_cg_version(self): return self.cg_version
  def get_vindex(self): return self.vindex

  def __repr__(self): return str(self)
  def __str__(self): return '%s([%s], %s, %s/%s)' % (self.name, ', '.join(map(str,self.args)), self.dim, self.vindex, self.cg_version)  #"expression %s/%s" % (self.vindex, self.cg_version)
  def __getitem__(self, i): return lookup(self, i)
  def __getslice__(self, i, j): return None
  def scalar_value(self, recalculate=False): return 0.0
  def vec_value(self, recalculate=False): return []
  def npvalue(self, recalculate=False): return None
  def value(self, recalculate=False): return None
  def forward(self, recalculate=False): return None
  def set(self, x): pass
  def batch(self, i): return lookup_batch(self, i)
  def zero(self): return self

  def backward(self): pass

  def __add__(self, other):
      if isinstance(self, Expression) and isinstance(other, Expression):
          return _add(self,other)
      elif isinstance(self, (int,float)) or isinstance(other, (int,float)):
          return _cadd(self, other)
      else: raise NotImplementedError('self=%s, other=%s' % (self, other))
  def __mul__(self, other):
      if isinstance(self, Expression) and isinstance(other, Expression):
          return _mul(self,other)
      elif isinstance(self, (int,float)) or isinstance(other, (int,float)):
          return _cmul(self, other)
      else: raise NotImplementedError('self=%s, other=%s' % (self, other))
  def __div__(self, other):
      if isinstance(self, Expression) and isinstance(other, (int,float)):
          return _cdiv(self, other)
      else: raise NotImplementedError()
  def __neg__(self):        return _neg(self)
  def __sub__(self, other):
      if isinstance(self,Expression) and isinstance(other,Expression):
          return self+(-other)
      elif isinstance(self,(int,float)) and isinstance(other,Expression):
          return _scalarsub(self, other)
      elif isinstance(self,Expression) and isinstance(other,(int, float)):
          return _neg(_scalarsub(other, self))
      else: raise NotImplementedError()
  def init_row(self, i, row): pass
  def init_from_array(self, *args, **kwargs): pass
  def set_updated(self, *args, **kwargs): pass


def GVExpr(name, args, dim): 
  e = Expression(name, args, dim)
  graphviz_items.append(e)
  return e


class Model(object):
    def add_parameters(self, dim, scale=0, *args, **kwargs):
        assert(isinstance(dim,(tuple,int)))
        pp = Expression('parameters', [dim], make_dim(dim))
        return pp

    def add_lookup_parameters(self, dim, *args, **kwargs):
        assert(isinstance(dim, tuple))
        pp = Expression('lookup_parameters', [dim], make_dim(dim[1]))
        return pp

    def save_all(self, fname): pass
    def load_all(self, fname): pass
    def save(self, fname): pass
    def load(self, fname): pass


SECRET = 923148
#_cg = ComputationGraph(SECRET)

def cg_version(): return _cg._cg_version
def renew_cg(immediate_compute=False, check_validity=False): return _cg.renew(immediate_compute, check_validity)

def cg():
    global _cg
    return _cg

class ComputationGraph(object):
    def __init__(self, guard=0):
        if guard != SECRET: raise RuntimeError("Do not instantiate ComputationGraph directly. Use pydynet.cg()")
        self._cg_version = 0

    def renew(self, immediate_compute=False, check_validity=False):
      vindex_count = -1
      del graphviz_items[:]
      return self

    def version(self): return self._cg_version

    def parameters(self, params):
      graphviz_items.append(params)
      return params

    def forward_scalar(self): return 0.0
    def inc_forward_scalar(self): return 0.0
    def forward_vec(self): return []
    def inc_forward_vec(self): return []
    def forward(self): return None
    def inc_forward(self): return None
    def backward(self): return None
_cg = ComputationGraph(SECRET)
# }}}



def parameter(p):
  graphviz_items.append(p)
  return p

def scalarInput(s): return GVExpr('scalarInput', [s], make_dim(1, inferred=True))
def vecInput(dim): return GVExpr('vecInput', [dim], make_dim(dim))
def inputVector(v): return GVExpr('inputVector', [v], make_dim(len(v), inferred=True))
def matInput(d1, d2): return GVExpr('matInput', [d1, d2], make_dim(d1, d2))
def inputMatrix(v, d): return GVExpr('inputMatrix', [v, d], make_dim(d, inferred=True))
def lookup(p, index=0, update=True): return GVExpr('lookup', [p, index, update], p.dim)
def lookup_batch(p, indices, update=True): return GVExpr('lookup_batch', [p, indices, update], p.dim)
def pick(a, index=0, dim=0): return GVExpr('pick', [a, index], make_dim(1, inferred=True))
def pick_batch(a, indices, dim=0): return GVExpr('pick_batch', [a, indices], make_dim(len(indices), inferred=True))
def hinge(x, index, m=1.0): return GVExpr('hinge', [x, index, m], copy_dim(x))
def max_dim(a, d=0): return GVExpr('max_dim', [a, d], make_dim(1, inferred=True))
def min_dim(a, d=0): return GVExpr('min_dim', [a, d], make_dim(1, inferred=True))

def nobackprop(x): return GVExpr('nobackprop', [x], copy_dim(x))
def flip_gradient(x): return GVExpr('flip_gradient', [x], copy_dim(x))

# binary-exp
def cdiv(x, y): return GVExpr('cdiv', [x,y], ensure_same_dim(x,y))
def colwise_add(x, y):
  if x.dim.invalid() or y.dim.invalid():
    d = InvalidDim
  elif x.dim[0] == y.dim[0] and y.dim[1] == 1:
    d = copy_dim(x)
  else:
    d = InvalidConcreteDim(x.dim, y.dim)
  return GVExpr('colwise_add', [x,y], d)

def trace_of_product(x, y): return GVExpr('trace_of_product', [x,y], ensure_same_dim(x,y))
def cmult(x, y): return GVExpr('cmult', [x,y], ensure_same_dim(x,y))
def dot_product(x, y): return GVExpr('dot_product', [x,y], ensure_same_dim(x,y))
def squared_distance(x, y): return GVExpr('squared_distance', [x,y], ensure_same_dim(x,y))
def l1_distance(x, y): return GVExpr('l1_distance', [x,y], ensure_same_dim(x,y))
def binary_log_loss(x, y): return GVExpr('binary_log_loss', [x,y], ensure_same_dim(x,y))
#def conv1d_narrow(x, y):
#  if x.dim.invalid() or y.dim.invalid():
#    d = InvalidDim
#  elif x.dim[0] != y.dim[0]:
#    d = InvalidConcreteDim(x.dim, y.dim)
#  else:
#    d = make_dim(x.dim[0], x.dim[1] - y.dim[1] + 1)
#  return GVExpr('conv1d_narrow', [x,y], d)
#def conv1d_wide(x, y):
#  if x.dim.invalid() or y.dim.invalid():
#    d = InvalidDim
#  elif x.dim[0] != y.dim[0]:
#    d = InvalidConcreteDim(x.dim, y.dim)
#  else:
#    d = make_dim(x.dim[0], x.dim[1] + y.dim[1] - 1)
#  return GVExpr('conv1d_wide', [x,y], d)
def filter1d_narrow(x, y): 
  if x.dim.invalid() or y.dim.invalid():
    d = InvalidDim
  elif x.dim[0] != y.dim[0]:
    d = InvalidConcreteDim(x.dim, y.dim)
  else:
    d = make_dim(x.dim[0], x.dim[1] - y.dim[1] + 1)
  return GVExpr('filter1d_narrow', [x,y], d)

# unary-exp
def tanh(x): return GVExpr('tanh', [x], copy_dim(x))
def exp(x): return GVExpr('exp', [x], copy_dim(x))
def square(x): return GVExpr('square', [x], copy_dim(x))
def sqrt(x): return GVExpr('sqrt', [x], copy_dim(x))
def erf(x): return GVExpr('erf', [x], copy_dim(x))
def cube(x): return GVExpr('cube', [x], copy_dim(x))
def log(x): return GVExpr('log', [x], copy_dim(x))
def lgamma(x): return GVExpr('lgamma', [x], copy_dim(x))
def logistic(x): return GVExpr('logistic', [x], copy_dim(x))
def rectify(x): return GVExpr('rectify', [x], copy_dim(x))
def log_softmax(x, restrict=None): return GVExpr('log_softmax', [x,restrict], copy_dim(x))
def softmax(x): return GVExpr('softmax', [x], copy_dim(x))
def softsign(x): return GVExpr('softsign', [x], copy_dim(x))
def pow(x, y): return GVExpr('pow', [x,y], ensure_same_dim(x,y))
def bmin(x, y): return GVExpr('bmin', [x,y], ensure_same_dim(x,y))
def bmax(x, y): return GVExpr('bmax', [x,y], ensure_same_dim(x,y))
def transpose(x): return GVExpr('transpose', [x], make_dim(x.dim[1], x.dim[0]) if x.dim.isvalid() else InvalidDim)
def sum_cols(x): return GVExpr('sum_cols', [x], make_dim(x.dim[0],1) if x.dim.isvalid() else InvalidDim)

def sum_batches(x): return GVExpr('sum_batches', [x], copy_dim(x))

#expr-opt
def fold_rows(x, nrows=2):
  if x.dim.invalid():
    d = InvalidDim
  elif x.dim[0] != nrows:
    d = InvalidConcreteDim(x.dim, nrows)
  else:
    d = make_dim(1, x.dim[1])
  return GVExpr('fold_rows', [x,nrows], d)
def pairwise_rank_loss(x, y, m=1.0): return GVExpr('pairwise_rank_loss', [x,y,m], ensure_same_dim(x,y))
def poisson_loss(x, y): return GVExpr('poisson_loss', [x,y], copy_dim(x))
def huber_distance(x, y, c=1.345): return GVExpr('huber_distance', [x,y,c], ensure_same_dim(x,y))
#expr-unsigned
def kmax_pooling(x, k, d=1): return GVExpr('kmax_pooling', [x,k,d], make_dim(x.dim[0], k) if x.dim.isvalid() else InvalidDim)
def pickneglogsoftmax(x, v): return GVExpr('pickneglogsoftmax', [x,v], make_dim(1, inferred=True))
def pickneglogsoftmax_batch(x, vs): return GVExpr('pickneglogsoftmax_batch', [x,vs], make_dim(len(vs), inferred=True))

def kmh_ngram(x, n): return GVExpr('kmh_ngram', [x,n], make_dim(x.dim[0], x.dim[1]-n+1) if x.dim.isvalid() else InvalidDim)
def pickrange(x, v, u): return GVExpr('pickrange', [x,v,u], make_dim(u-v, x.dim[1]) if x.dim.isvalid() else InvalidDim)
#expr-float
def noise(x, stddev): return GVExpr('noise', [x,stddev], copy_dim(x))
def dropout(x, p): return GVExpr('dropout', [x,p], copy_dim(x))
def block_dropout(x, p): return GVExpr('block_dropout', [x,p], copy_dim(x))
#expr-dim
def reshape(x, d): return GVExpr('reshape', [x,d], make_dim(d))
def esum(xs): return GVExpr('esum', xs, ensure_all_same_dim(xs))
def average(xs): return GVExpr('average', xs, ensure_all_same_dim(xs))
def emax(xs): return GVExpr('emax', xs, ensure_all_same_dim(xs))
def concatenate_cols(xs):
  if any(x.dim.invalid() for x in xs):
    dim = InvalidDim
  else:
    nrows = xs[0].dim[0]
    ncols = xs[0].dim[1]
    for x in xs:
      ncols += x.dim[1]
      nrows = nrows if nrows == x.dim[0] else -1
    dim = make_dim(nrows, ncols) if nrows >= 0 else InvalidDim
  return GVExpr('concatenate_cols', xs, dim)
def concatenate(xs):
  if any(x.dim.invalid() for x in xs):
    dim = InvalidDim
  else: 
    nrows = xs[0].dim[0]
    ncols = xs[0].dim[1]
    for x in xs[1:]:
      nrows += x.dim[0]
      ncols = ncols if ncols == x.dim[1] else -1
    dim = make_dim(nrows, ncols) if ncols >= 0 else InvalidDim
  return GVExpr('concatenate', xs, dim)

def affine_transform(xs):
  if any(x.dim.invalid() for x in xs):
    dim = InvalidDim
  elif all(ensure_mul_dim(a,b)==xs[0].dim for a,b in zip(xs[1::2],xs[2::2])):
    dim = xs[0].dim
  else:
    dim = InvalidDim
  return GVExpr('affine_transform', xs, dim)










builder_num = -1
def new_builder_num():
  global builder_num
  builder_num += 1
  return builder_num

class _RNNBuilder(object):
  def set_dropout(self, f): pass
  def disable_dropout(self): pass

  def new_graph(self):
      self.cg_version = _cg.version()
      self.builder_version = new_builder_num()

  def start_new_sequence(self, es=None):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")

  def add_input(self, e):
      ensure_freshness(e)
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      return Expression.from_cexpr(self.cg_version, self.thisptr.add_input(e.c()))

  def add_input_to_prev(self, prev, e):
      ensure_freshness(e)
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      return Expression.from_cexpr(self.cg_version, self.thisptr.add_input(prev, e.c()))

  def rewind_one_step(self):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      self.thisptr.rewind_one_step()

  def back(self):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      return Expression.from_cexpr(self.cg_version, self.thisptr.back())

  def final_h(self):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      res = []
      #def CExpression cexp
      cexps = self.thisptr.final_h()
      for cexp in cexps:
          res.append(Expression.from_cexpr(self.cg_version, cexp))
      return res

  def final_s(self):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      res = []
      #def CExpression cexp
      cexps = self.thisptr.final_s()
      for cexp in cexps:
          res.append(Expression.from_cexpr(self.cg_version, cexp))
      return res

  def get_h(self, i):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      res = []
      #def CExpression cexp
      cexps = self.thisptr.get_h(i)
      for cexp in cexps:
          res.append(Expression.from_cexpr(self.cg_version, cexp))
      return res

  def get_s(self, i):
      if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
      res = []
      #def CExpression cexp
      cexps = self.thisptr.get_s(i)
      for cexp in cexps:
          res.append(Expression.from_cexpr(self.cg_version, cexp))
      return res

  def initial_state(self,vecs=None):
      if self._init_state is None or self.cg_version != _cg.version():
          self.new_graph()
          if vecs is not None:
              self.start_new_sequence(vecs)
          else:
              self.start_new_sequence()
          self._init_state = RNNState(self, -1)
      return self._init_state

  def initial_state_from_raw_vectors(self,vecs=None):
      if self._init_state is None or self.cg_version != _cg.version():
          self.new_graph()
          if vecs is not None:
              es = []
              for v in vecs:
                  e = vecInput(len(v))
                  e.set(v)
                  es.append(e)
              self.start_new_sequence(es)
          else:
              self.start_new_sequence()
          self._init_state = RNNState(self, -1)
      return self._init_state

class SimpleRNNBuilder(_RNNBuilder):
  def __init__(self, layers, input_dim, hidden_dim, model): 
    self.cg_version = -1
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.model = model
    self._init_state = None
    self.builder_version = new_builder_num()
  def whoami(self): return "SimpleRNNBuilder"
class GRUBuilder(_RNNBuilder):
  def __init__(self, layers, input_dim, hidden_dim, model): 
    self.cg_version = -1
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.model = model
    self._init_state = None
    self.builder_version = new_builder_num()
  def whoami(self): return "GRUBuilder"
class LSTMBuilder(_RNNBuilder):
  def __init__(self, layers, input_dim, hidden_dim, model): 
    self.cg_version = -1
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.model = model
    self._init_state = None
    self.builder_version = new_builder_num()
  def whoami(self): return "LSTMBuilder"
class FastLSTMBuilder(_RNNBuilder):
  def __init__(self, layers, input_dim, hidden_dim, model): 
    self.cg_version = -1
    self.layers = layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.model = model
    self._init_state = None
    self.builder_version = new_builder_num()
  def whoami(self): return "FastLSTMBuilder"

class BiRNNBuilder(object):
    """
    Builder for BiRNNs that delegates to regular RNNs and wires them together.  
    
        builder = BiRNNBuilder(1, 128, 100, model, LSTMBuilder)
        [o1,o2,o3] = builder.transduce([i1,i2,i3])
    """
    def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory):
        """
        @param num_layers: depth of the BiRNN
        @param input_dim: size of the inputs
        @param hidden_dim: size of the outputs (and intermediate layer representations)
        @param model
        @param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
        """
        assert num_layers > 0
        assert hidden_dim % 2 == 0
        self.builder_layers = []
        f = rnn_builder_factory(1, input_dim, hidden_dim/2, model)
        b = rnn_builder_factory(1, input_dim, hidden_dim/2, model)
        self.builder_layers.append((f,b))
        for _ in xrange(num_layers-1):
            f = rnn_builder_factory(1, hidden_dim, hidden_dim/2, model)
            b = rnn_builder_factory(1, hidden_dim, hidden_dim/2, model)
            self.builder_layers.append((f,b))

    def whoami(self): return "BiRNNBuilder"

    def set_dropout(self, p):
      for (fb,bb) in self.builder_layers:
        fb.set_dropout(p)
        bb.set_dropout(p)
    def disable_dropout(self):
      for (fb,bb) in self.builder_layers:
        fb.disable_dropout()
        bb.disable_dropout()

    def add_inputs(self, es):
        """
        returns the list of state pairs (stateF, stateB) obtained by adding 
        inputs to both forward (stateF) and backward (stateB) RNNs.  

        @param es: a list of Expression

        see also transduce(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        for e in es:
            ensure_freshness(e)
        for (fb,bb) in self.builder_layers[:-1]:
            fs = fb.initial_state().transduce(es)
            bs = bb.initial_state().transduce(reversed(es))
            es = [concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        (fb,bb) = self.builder_layers[-1]
        fs = fb.initial_state().add_inputs(es)
        bs = bb.initial_state().add_inputs(reversed(es))
        return [(f,b) for f,b in zip(fs, reversed(bs))]

    def transduce(self, es):
        """
        returns the list of output Expressions obtained by adding the given inputs
        to the current state, one by one, to both the forward and backward RNNs, 
        and concatenating.
        
        @param es: a list of Expression

        see also add_inputs(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        for e in es:
            ensure_freshness(e)
        for (fb,bb) in self.builder_layers:
            fs = fb.initial_state().transduce(es)
            bs = bb.initial_state().transduce(reversed(es))
            es = [concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        return es

class RNNState(object): # {{{
    def __init__(self, builder, state_idx=-1, prev_state=None, out=None):
      self.builder = builder
      self.state_idx=state_idx
      self._prev = prev_state
      self._out = out

    def add_input(self, x): # x: Expression
      input_dim = make_dim(self.builder.input_dim)
      input_dim = x.dim if x.dim==input_dim else InvalidConcreteDim(x.dim, input_dim)
      rnn_type = self.builder.whoami()
      if rnn_type.endswith("Builder"): rnn_type = rnn_type[:-len("Builder")]
      output_e = GVExpr('RNNState', [x, input_dim, rnn_type, self.builder.builder_version, self.state_idx+1], dim=make_dim(self.builder.hidden_dim))
      new_state = RNNState(self.builder, self.state_idx+1, self, output_e)
      return new_state

    def add_inputs(self, xs):
        if self._prev is None:
          self.builder.builder_version = new_builder_num()
        states = []
        cur = self
        for x in xs:
            cur = cur.add_input(x)
            states.append(cur)
        return states
    
    def transduce(self, xs):
        return [x.output() for x in self.add_inputs(xs)]

    def output(self): return self._out

    def prev(self): return self._prev
    def b(self): return self.builder
    def get_state_idx(self): return self.state_idx

# StackedRNNState   TODO: do at least minimal testing for this #{{{
class StackedRNNState(object):
    #def list states
    #def StackedRNNState prev
    def __init__(self, states, prev=None):
        self.states = states
        self.prev = prev

    def add_input(self, x):
        #def next_states
        next_states = []
        for s in self.states:
            next_states.append(s.add_input(x))
            x = next_states[-1].output()
        return StackedRNNState(next_states, self)

    def output(self): return self.states[-1].output()

    def prev(self): return self.prev
    def h(self): return [s.h() for s in self.states]
    def s(self): return [s.s() for s in self.states]

    def add_inputs(self, xs):
        """
        returns the list of states obtained by adding the given inputs
        to the current state, one by one.
        """
        states = []
        cur = self
        for x in xs:
            cur = cur.add_input(x)
            states.append(cur)
        return states


class Trainer(object):
  def update(self, s=1.0): pass
  def update_epoch(self, r = 1.0): pass
  def status(self): pass
  def set_clip_threshold(self, thr): pass
  def get_clip_threshold(self): pass

class SimpleSGDTrainer(Trainer):
    """
    This object is very cool!
    """
    def __init__(self, m, e0 = 0.1, *args): pass
class MomentumSGDTrainer(Trainer):
    def __init__(self, m, e0 = 0.01, mom = 0.9, *args): pass
class AdagradTrainer(Trainer):
    def __init__(self, m, e0 = 0.1, eps = 1e-20, *args): pass
class AdadeltaTrainer(Trainer):
    def __init__(self, m, eps = 1e-6, rho = 0.95, *args): pass
class AdamTrainer(Trainer):
    def __init__(self, m, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8, *args ): pass


class Initializer(object): pass
class NormalInitializer(Initializer):
    def __init__(self, mean=0, var=1): pass
class UniformInitializer(Initializer):
    def __init__(self, scale): pass
class ConstInitializer(Initializer):
    def __init__(self, c): pass
class GlorotInitializer(Initializer):
    def __init__(self, is_lookup=False): pass
class FromFileInitializer(Initializer):
    def __init__(self, fname): pass
class NumpyInitializer(Initializer):
    def __init__(self, array): pass
















def shape_str(e_dim):
  if e_dim.invalid():
    #return '{??}'
    return str(e_dim)
  elif e_dim.inferred:
    if e_dim[1] == 1:
      return '{%s}' % (e_dim[0])
    else:
      return '{%s,%s}' % (e_dim[0],e_dim[1])
  else:
    if e_dim[1] == 1:
      return '{{%s}}' % (e_dim[0])
    else:
      return '{{%s,%s}}' % (e_dim[0],e_dim[1])

class GVNode(object):
  def __init__(self, name, input_dim, label, output_dim, children, features, node_type, expr_name):
    self.name = name
    self.input_dim = input_dim
    self.label = label
    self.output_dim = output_dim
    self.children = children
    self.features = features
    self.node_type = node_type
    self.expr_name = expr_name
  def __iter__(self): return iter([self.name, self.input_dim, self.label, self.output_dim, self.children, self.features, self.node_type, self.expr_name])
  def __repr__(self): return 'GVNode(%s)' % ', '.join(map(str, self))
  def __str__(self): return repr(self)
  def __lt__(self, other): return id(self) < id(other)

def make_network_graph(compact, expression_names, lookup_names):
  """
  Make a network graph, represented as of nodes and a set of edges.  
  The nodes are represented as tuples: (name: string, input_dim: Dim, label: string, output_dim: Dim, children: set[name], features: string)
#   The edges are represented as dict of children to sets of parents: (child: string) -> [(parent: string, features: string)] 
  """
  nodes = set()
#   edges = defaultdict(set) # parent -> (child, extra)
  
  var_name_dict = dict()
  if expression_names:
    for e in graphviz_items: # e: Expression
      if e in expression_names:
        var_name_dict[e.vindex] = expression_names[e]
  
  rnn_bldr_name = defaultdict(lambda: chr(len(rnn_bldr_name)+ord('A')))
  def vidx2str(vidx): return '%s%s' % ('N', vidx)

  for e in graphviz_items: # e: Expression
    vidx = e.vindex
    f_name = e.name
    args = e.args
    output_dim = e.dim
    input_dim = None # basically just RNNStates use this since everything else has input_dim==output_dim
    children = set()
    node_type = '2_regular'
    
    if f_name == 'vecInput':
      [_dim] = args
      arg_strs = []
    elif f_name == 'inputVector':
      [_v] = args
      arg_strs = []
    elif f_name == 'matInput':
      [_d1, _d2] = args
      arg_strs = []
    elif f_name == 'inputMatrix':
      [_v, _d] = args
      arg_strs = []
    elif f_name == 'parameters':
      [_dim] = args
      arg_strs = []
      if compact:
        if vidx in var_name_dict:
          f_name = var_name_dict[vidx]
      node_type = '1_param'
    elif f_name == 'lookup_parameters':
      [_dim] = args
      arg_strs = []
      if compact:
        if vidx in var_name_dict:
          f_name = var_name_dict[vidx]
      node_type = '1_param'
    elif f_name == 'lookup':
      [p, idx, update] = args
      [_dim] = p.args
      if vidx in var_name_dict:
        name = var_name_dict[vidx]
      else:
        name = None
      item_name = None
      if lookup_names and p in expression_names:
        param_name = expression_names[p]
        if param_name in lookup_names:
          item_name = '\\"%s\\"' % (lookup_names[param_name][idx],)
      if compact:
        if item_name is not None:
          f_name = item_name
        elif name is not None:
          f_name = '%s[%s]' % (name, idx)
        else:
          f_name = 'lookup(%s)' % (idx)
        arg_strs = []
      else:
        arg_strs = [var_name_dict.get(p.vindex, 'v%d' % (p.vindex))]
        if item_name is not None:
          arg_strs.append(item_name)
        vocab_size = _dim[0]
        arg_strs.extend(['%s' % (idx), '%s' % (vocab_size), 'update' if update else 'fixed'])
      #children.add(vidx2str(p.vindex))
      #node_type = '1_param'
    elif f_name == 'RNNState':
      [arg, input_dim, bldr_type, bldr_num, state_idx] = args # arg==input_e
      rnn_name = rnn_bldr_name[bldr_num]
      if bldr_type.endswith('Builder'):
        bldr_type[:-len('Builder')]
      f_name = '%s-%s-%s' % (bldr_type, rnn_name, state_idx)
      if not compact:
        i = arg.vindex
        s = var_name_dict.get(i, 'v%d' % (i))
        arg_strs = [s]
      else:
        arg_strs = []
      children.add(vidx2str(arg.vindex))
      node_type = '3_rnn_state'
    else:
      arg_strs = []
      for arg in args:
        if isinstance(arg, Expression):
          if not compact:
            i = arg.vindex
            s = var_name_dict.get(i, 'v%d' % (i))
            arg_strs.append(s)
          children.add(vidx2str(arg.vindex))
        elif isinstance(arg, float) and compact:
          s = re.sub('0+$', '', '%.3f' % (arg))
          if s == '0.':
            s = str(arg)
          arg_strs.append(s)
        else:
          arg_strs.append(str(arg))
        
#     f_name = { ,
#              }.get(f_name, f_name)
      
    if compact:
      f_name = { 'add': '+',
                 'sub': '-',
                 'mul': '*',
                 'div': '/',
                 'cadd': '+',
                 'cmul': '*',
                 'cdiv': '/',
                 'scalarsub': '-',
                 'concatenate': 'cat',
                 'esum': 'sum',
                 'emax': 'max',
                 'emin': 'min',
               }.get(f_name, f_name)
      if arg_strs:
        str_repr = '%s(%s)' % (f_name, ', '.join(arg_strs))
      else:
        str_repr = f_name
    elif f_name == 'add':
      [a,b] = arg_strs
      str_repr = '%s + %s' % (a,b)
    elif f_name == 'sub':
      [a,b] = arg_strs
      str_repr = '%s - %s' % (a,b)
    elif f_name == 'mul':
      [a,b] = arg_strs
      str_repr = '%s * %s' % (a,b)
    elif f_name == 'div':
      [a,b] = arg_strs
      str_repr = '%s / %s' % (a,b)
    elif f_name == 'neg':
      [a,] = arg_strs
      str_repr = '-%s' % (a)
    elif f_name == 'affine_transform':
      str_repr = arg_strs[0]
      for i in xrange(1, len(arg_strs), 2):
        str_repr += ' + %s*%s' % tuple(arg_strs[i:i+2])
    else:
      if arg_strs is not None:
        str_repr = '%s(%s)' % (f_name, ', '.join(arg_strs))
      else:
        str_repr = f_name
        
    name = vidx2str(vidx)
    var_name = '%s' % (var_name_dict.get(vidx, 'v%d' % (vidx))) if not compact else ''
#     if show_dims:
#       str_repr = '%s\\n%s' % (shape_str(e.dim), str_repr)
    label = str_repr
    if not compact:
      label = '%s = %s' % (var_name, label)
    features = ''
#     if output_dim.invalid():
#       features += " [color=red,style=filled,fillcolor=red]"
#     node_def_lines.append('  %s [label="%s%s"] %s;' % (vidx2str(vidx), label_prefix, str_repr, ''))
    expr_name = expression_names[e] if compact and expression_names and (e in expression_names) and (expression_names[e] != f_name) else None
    nodes.add(GVNode(name, input_dim, label, output_dim, frozenset(children), features, node_type, expr_name))

  return nodes

def parents_of(n, nodes):
  ps = []
  for n in nodes:
    for c in n.children:
      if n in c.children:
        ps.append
  return ps

def collapse_birnn_states(nodes, compact):
  node_info = {n.name:n for n in nodes}
  new_nodes = []
  children_forwards = dict()  # if `n.children` is pointing to K, return V instead
  rnn_state_nodes = []
  rnn_parents = defaultdict(set) # rnn_state_node -> [parent_expression]
  rnn_children = {}              # rnn_state_node -> [child_expression]
  shared_rnn_states = defaultdict(set) # (input name, output name) -> [(rnn state name)]
  rnn_groups = dict() # these nodes (keys) are being replaced by the new group nodes (values)
  nodes_to_delete = set()
  for n in nodes:
    for c in n.children:
      if node_info[c].node_type == '3_rnn_state':
        rnn_parents[node_info[c].name].add(n.name)
    if n.node_type == '3_rnn_state':
      rnn_state_nodes.append(n)
      rnn_children[n.name] = set(node_info[c].name for c in n.children)
  for n in rnn_state_nodes:
    in_e, = rnn_children[n.name]
    out_e, = rnn_parents[n.name]
    shared_rnn_states[(in_e, out_e)].add(n)
  for ((in_e, out_e), ns) in shared_rnn_states.iteritems():
    input_dims = set(n.input_dim for n in ns)
    output_dims = set(n.output_dim for n in ns)
    if len(ns) > 1 and len(input_dims)==1 and len(output_dims)==1:
      input_dim, = input_dims
      output_dim, = output_dims
      new_rnn_group_state_name = ''.join(n.name for n in sorted(ns))
      new_rnn_group_state_label = '\\n'.join(n.label for n in sorted(ns))
      if not compact:
        new_rnn_group_state_label = '%s\\n%s' % (node_info[out_e].label, new_rnn_group_state_label)
      cat_output_dim = make_dim(output_dim[0]*2, output_dim[1])
      new_rnn_group_state = GVNode(new_rnn_group_state_name, input_dim, new_rnn_group_state_label, cat_output_dim, frozenset([in_e]), '', '3_rnn_state', node_info[out_e].expr_name)
      for n in ns:
        rnn_groups[n.name] = new_rnn_group_state.name
#         children_forwards[n.name] = new_rnn_group_state.name
        nodes_to_delete.add(n.name)
      children_forwards[out_e] = new_rnn_group_state.name
      nodes.add(new_rnn_group_state)
      nodes_to_delete.add(out_e)
  # TODO: WHEN WE DELETE A CAT NODE, MAKE SURE WE FORWARD TO THE **NEW GROUP STATE NODE**
  for (name, input_dim, label, output_dim, children, features, node_type, expr_name) in nodes:
    if name not in nodes_to_delete:
      new_children = []
      for c in children:
        while c in children_forwards:
          c = children_forwards[c]
        new_children.append(c)
      new_nodes.append(GVNode(name, input_dim, label, output_dim, new_children, features, node_type, expr_name))
  return (new_nodes, rnn_groups)

def print_graphviz(compact=False, show_dims=True, expression_names=None, lookup_names=None, collapse_birnns=False):
  original_nodes = make_network_graph(compact, expression_names, lookup_names)
  nodes = original_nodes
  collapse_to = dict()
  if collapse_birnns:
    (nodes, birnn_collapse_to) = collapse_birnn_states(nodes, compact)
    collapse_to.update(birnn_collapse_to)

  print('digraph G {')
  print('  rankdir=BT;')
  if not compact: print('  nodesep=.05;')
  
  node_types = defaultdict(set)
  for n in nodes:
    node_types[n.node_type].add(n.name)
  for node_type in sorted(node_types):
    style = {
              '1_param': '[shape=ellipse]',
              '2_regular': '[shape=rect]',
              '3_rnn_state': '[shape=rect, peripheries=2]',
             }[node_type]
    print('  node %s; ' % (style), ' '.join(node_types[node_type]))
  
#   all_nodes = set(line.strip().split()[0] for line in node_def_lines)
  for n in nodes:
    label = n.label
    if show_dims:
      if n.expr_name is not None:
        label = '%s\\n%s' % (n.expr_name, label)
      label = '%s\\n%s' % (shape_str(n.output_dim), label)
      if n.input_dim is not None:
        label = '%s\\n%s' % (label, shape_str(n.input_dim))
    if n.output_dim.invalid() or (n.input_dim is not None and n.input_dim.invalid()):
      n.features += " [color=red,style=filled,fillcolor=red]"
    print('  %s [label="%s"] %s;' % (n.name, label, n.features))
    for c in n.children:
      print('  %s -> %s;' % (c, n.name))
    
  rnn_states = [] # (name, rnn_name, state_idx)
  rnn_state_re = re.compile("[^-]+-(.)-(\\d+)")
  for n in original_nodes:
    if n.node_type == '3_rnn_state':
      m = rnn_state_re.search(n.label)
      assert m is not None, 'rnn_state_re.search(%s); %s' % (n.label, n)
      (rnn_name, state_idx) = m.groups()
      rnn_states.append((rnn_name, int(state_idx), n.name))
  rnn_states = sorted(rnn_states)
  edges = set()
  for ((rnn_name_p, state_idx_p, name_p), (rnn_name_n, state_idx_n, name_n)) in zip(rnn_states,rnn_states[1:]):
    if rnn_name_p == rnn_name_n:
      if state_idx_p+1 == state_idx_n:
        group_name_p = collapse_to.get(name_p, name_p)
        group_name_n = collapse_to.get(name_n, name_n)
        edges.add((group_name_p, group_name_n))
  for (name_p, name_n) in edges:
    print('  %s -> %s [style=dotted];' % (name_p, name_n)) # ,dir=both

  print('}')
