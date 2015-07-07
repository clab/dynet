import sys
from cython.operator cimport dereference as deref
# TODO:
#  - set random seed (in CNN)
#  - better input / output support
#    WORKS, but need to be unified? for example, why "pick" takes a pointer to int, and "squared_distance" takes an expression?
#  - load embeddings file
#  - load/save models
#  - make ComputationGraph a singleton???
#  - NOTE: why do we need to filter short sentences in rnnlm.py or crash??

# Examples:
#  V xor  
#  V xor-xent
#  - textcat
#  - tag-bilstm
#  - rnnlm
#  V nlm
#  - encdec
#  - embedcl
#  - embed/nlm -- negative sampling?

from pycnn cimport *
cimport pycnn

cdef init():
    cdef char** argv = []
    cdef int argc = 0
    pycnn.Initialize(argc,argv)
init()

cdef CDim Dim(dim):
    """
    dim: either a tuple or an int
    """
    if isinstance(dim, tuple):
        if len(dim) == 1: return CDim(dim[0])
        elif len(dim) == 2: return CDim(dim[0],dim[1])
        else:
            raise "Unsupported dimension",dim
    # hope it's a number. TODO: error checking / exception
    if isinstance(dim, int):
        return CDim(dim)
    raise "Unsupported dimension",dim

cdef class Parameters:
    cdef CParameters *thisptr
    def __cinit__(self):
        self.thisptr = NULL
    @staticmethod
    cdef wrap_ptr(CParameters* ptr):
        self = Parameters()
        self.thisptr = ptr
        return self

cdef class LookupParameters:
    cdef CLookupParameters *thisptr
    def __cinit__(self):
        self.thisptr = NULL
    @staticmethod
    cdef wrap_ptr(CLookupParameters* ptr):
        self = LookupParameters()
        self.thisptr = ptr
        return self

cdef class Model:
    cdef CModel *thisptr
    cdef object named_params
    def __cinit__(self):
        self.thisptr = new CModel()
    def __init__(self):
        self.named_params = {}

    def __dealloc__(self): del self.thisptr

    def add_parameters(self, name, dim, scale=0):
        cdef CParameters* p
        p = self.thisptr.add_parameters(Dim(dim))
        cdef Parameters pp = Parameters.wrap_ptr(p)
        self.named_params[name] = pp
        return pp

    def add_lookup_parameters(self, name, dim, scale=0):
        assert(isinstance(dim, tuple))
        cdef int nids = dim[0]
        rest = tuple(dim[1:])
        cdef CLookupParameters* p
        p = self.thisptr.add_lookup_parameters(nids, Dim(rest))
        cdef LookupParameters pp = LookupParameters.wrap_ptr(p)
        self.named_params[name] = pp
        return pp

    def __getitem__(self, name):
        return self.named_params[name]

    def __contains__(self, name):
        return name in self.named_params



# {{{ Computation Graph 

# {{{ "Pointers"

cdef class UnsignedValue:
    cdef unsigned val
    def __cinit__(self, val = 0): self.val = val
    def set(self, val): self.val = val
    def get(self): return self.val
    cdef unsigned* addr(self): return &(self.val)

cdef class FloatValue:
    cdef float val
    def __cinit__(self, val = 0): self.val = val
    def set(self, val): self.val = val
    def get(self): return self.val
    cdef float* addr(self): return &(self.val)

cdef class FloatVectorValue:
    cdef vector[float] *vals
    def __cinit__(self, vals):
        self.vals = new vector[float]()
        self.set(vals)
    def __dealloc__(self):
        del self.vals
    def set(self, newval):
        self.vals.clear()
        for f in newval: self.vals.push_back(f)
    def get(self): return deref(self.vals)
    def size(self): return len(deref(self.vals))
    cdef vector[float]* addr(self): return self.vals

# }}}

cdef class ComputationGraph:
    cdef CComputationGraph *thisptr, 
    cdef list _inputs
    def __cinit__(self):
        self.thisptr = new CComputationGraph()
        self._inputs = []
    def __dealloc__(self):
        del self.thisptr

    cpdef renew(self):
        del self.thisptr
        self.thisptr = new CComputationGraph()
        self._inputs = []
        return self

    def parameters(self, model, name, dim=None):
        cdef Parameters params
        cdef Expression result
        if name in model:
            params = model[name]
        else:
            assert(dim is not None)
            params = model.add_parameters(name, dim)
        result = Expression.from_cexpr(c_parameter(self.thisptr[0], params.thisptr))
        #result = self._expr(parameter(self.thisptr[0], params.thisptr))
        #result = Expression.wrap_ptr(&(parameter(self.thisptr[0], params.thisptr)))
        return result

    cpdef forward_scalar(self):
        return c_as_scalar(self.thisptr.forward())

    cpdef inc_forward_scalar(self):
        return c_as_scalar(self.thisptr.incremental_forward())

    cpdef forward_vec(self):
        return c_as_vector(self.thisptr.forward())

    cpdef inc_forward_vec(self):
        return c_as_vector(self.thisptr.incremental_forward())

    cpdef backward(self):
        self.thisptr.backward()

    cpdef PrintGraphviz(self):
        self.thisptr.PrintGraphviz()

    # CNN handles changing inputs keeping pointers to memoty locations.
    # Because of python's memory management, objects that wrap such pointers
    # must be registered in a central location. This location would be the
    # computation graph.
    #
    # We need to support the following cases:
    #   input(real*)
    #   input(vector<real>*)
    #   lookup(params, unsigned*)
    #   const_lookup(params, unsigned*)
    #   pick(unsigned*)
    # 
    # We have the classes UnsignedValue, FloatValue and FloatVectorValue for
    # this purpose.
    cpdef inputValue(self, FloatValue v):
        self._inputs.append(v)
        return inputValue(self, v)
    cpdef inputVector(self, FloatVectorValue fv):
        self._inputs.append(fv)
        return inputVector(self, fv)
    cpdef lookup(self, LookupParameters p, UnsignedValue v, update=True):
        self._inputs.append(v)
        return lookup(self, p, v, update)




# }}}

#{{{ Expressions

cdef _add(Expression a, Expression b): return Expression.from_cexpr(c_op_add(a.c(), b.c()))
cdef _mul(Expression a, Expression b): return Expression.from_cexpr(c_op_mul(a.c(), b.c()))
cdef _neg(Expression a): return Expression.from_cexpr(c_op_neg(a.c()))
cdef _scalarsub(float a, Expression b): return Expression.from_cexpr(c_op_scalar_sub(a, b.c()))
cdef _cmul(Expression a, float b): return Expression.from_cexpr(c_op_scalar_mul(a.c(), b))

cdef class Expression: #{{{
    cdef CComputationGraph *cg
    cdef VariableIndex vindex
    def __cinit__(self):
        self.cg = NULL
        self.vindex = 0
    @staticmethod
    cdef Expression from_cexpr(CExpression cexpr):
        self = Expression()
        self.cg = cexpr.pg
        self.vindex = cexpr.i
        return self
    cdef CExpression c(self):
        return CExpression(self.cg, self.vindex)

    def __repr__(self):
        return str(self)
    def __str__(self):
        return "exprssion %s" % <int>self.vindex
    #cdef scalar(self):
    #    return c_as_scalar(self.cg.get_value(self.vindex))

    def __add__(self, other): return _add(self,other)
    def __mul__(self, other):
        if isinstance(self, Expression) and isinstance(other, Expression):
            return _mul(self,other)
        elif isinstance(self, (int,float)) or isinstance(other, (int,float)):
            return _cmul(self, other)
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
#}}}

cpdef Expression parameter(ComputationGraph g, Parameters p):
    return Expression.from_cexpr(c_parameter(g.thisptr[0], p.thisptr))

# {{{ Mutable Expressions
#     These depend values that can be set by the caller

#cdef class InputExpression(Expression):
#    cdef float val
#    def __cinit__(self, ComputationGraph g, float s):
#        self.val = s
#        self.cg = g.thisptr
#        cdef CExpression e
#        e = c_input(self.cg[0], &(self.val))
#        self.vindex = e.i
#    def set(self, float s):
#        self.val = s

cdef class inputExpression(Expression):
    cdef FloatValue val
    def __cinit__(self, ComputationGraph g, float s):
        self.val = FloatValue(s)
        self.cg = g.thisptr
        cdef CExpression e
        e = c_input(self.cg[0], self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self, float s):
        self.val.set(s)
#
#cdef class VecInputExpression(Expression):
#    cdef vector[float] *data
#    def __cinit__(self, ComputationGraph g, vector[float] data):
#        cdef float f
#        dim = len(data)
#        self.data = new vector[float]()
#        for f in data:
#            self.data.push_back(f)
#        self.cg = g.thisptr
#        cdef CExpression e
#        e = c_input(self.cg[0], Dim(dim), self.data)
#        self.vindex = e.i
#    def __dealloc__(self):
#        print >> sys.stderr,"del"
#        del self.data
#    def set(self, vector[float] data):
#        cdef float f
#        self.data.clear()
#        for f in data:
#            self.data.push_back(f)

cdef class vecInputExpression(Expression):
    cdef FloatVectorValue val
    def __cinit__(self, ComputationGraph g, FloatVectorValue val):
        self.val = val
        self.cg = g.thisptr
        cdef CExpression e
        e = c_input(self.cg[0], Dim(val.size()), self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self, vector[float] data):
        self.val.set(data)

#cdef class LookupExpression(Expression):
#    cdef unsigned index
#    def __cinit__(self, ComputationGraph g, LookupParameters p, unsigned index=0):
#        self.index = 1
#        #self.pindex = &(self.index)
#        #self.pindex[0] = index
#        self.cg = g.thisptr
#        cdef CExpression e
#        e = c_lookup(self.cg[0], p.thisptr, &(self.index))
#        self.vindex = e.i
#    def set(self,i): self.index=i

cdef class lookupExpression(Expression):
    cdef UnsignedValue val
    def __cinit__(self, ComputationGraph g, LookupParameters p, unsigned index=0, update=True):
        self.val = UnsignedValue(index)
        self.cg = g.thisptr
        cdef CExpression e
        if update:
            e = c_lookup(self.cg[0], p.thisptr, self.val.addr())
        else:
            e = c_const_lookup(self.cg[0], p.thisptr, self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self,i): self.val.set(i)

#cdef class ConstLookupExpression(Expression):
#    """Const in the sense that the lookup table is not updated.
#    """
#    cdef unsigned index
#    def __cinit__(self, ComputationGraph g, LookupParameters p, unsigned index):
#        self.index = index
#        self.cg = g.thisptr
#        cdef CExpression e
#        e = c_const_lookup(self.cg[0], p.thisptr, &(self.index))
#        self.vindex = e.i
#    def set_index(self, unsigned index):
#        self.index = index
# }}}

#cdef class pick(Expression):
#    cdef IntValue index
#    def __cinit__(self, Expression x, IntValue index):
#        self.index = index
#        self.cg = x.cg
#        cdef CExpression e
#        e = c_pick(x.c(), <unsigned*>&(self.index.val))
#        self.vindex = e.i
#    #def set_index(self, unsigned i):
#    #    self.index = i


cdef Expression inputValue(ComputationGraph cg, FloatValue v):
    return Expression.from_cexpr(c_input(cg.thisptr[0], v.addr()))

cdef Expression inputVector(ComputationGraph cg, FloatVectorValue v):
    return Expression.from_cexpr(c_input(cg.thisptr[0], Dim(v.size()), v.addr()))

cdef Expression lookup(ComputationGraph cg, LookupParameters p, UnsignedValue v, update):
    if update:
        return Expression.from_cexpr(c_lookup(cg.thisptr[0], p.thisptr, v.addr()))
    else:
        return Expression.from_cexpr(c_const_lookup(cg.thisptr[0], p.thisptr, v.addr()))

cpdef Expression pick(Expression x, UnsignedValue v):
    # TODO register automatically in CG!! how?
    #x.cg._inputs.append(v)
    return Expression.from_cexpr(c_pick(x.c(), v.addr()))


# binary-exp
cpdef Expression cdiv(Expression x, Expression y): return Expression.from_cexpr(c_cdiv(x.c(), y.c()))
cpdef Expression colwise_add(Expression x, Expression y): return Expression.from_cexpr(c_colwise_add(x.c(), y.c()))

cpdef Expression cwise_multiply(Expression x, Expression y): return Expression.from_cexpr(c_cwise_multiply(x.c(), y.c()))
cpdef Expression dot_product(Expression x, Expression y): return Expression.from_cexpr(c_dot_product(x.c(), y.c()))
cpdef Expression squared_distance(Expression x, Expression y): return Expression.from_cexpr(c_squared_distance(x.c(), y.c()))
cpdef Expression l1_distance(Expression x, Expression y): return Expression.from_cexpr(c_l1_distance(x.c(), y.c()))
cpdef Expression binary_log_loss(Expression x, Expression y): return Expression.from_cexpr(c_binary_log_loss(x.c(), y.c()))
cpdef Expression conv1d_narrow(Expression x, Expression y): return Expression.from_cexpr(c_conv1d_narrow(x.c(), y.c()))
cpdef Expression conv1d_wide(Expression x, Expression y): return Expression.from_cexpr(c_conv1d_wide(x.c(), y.c()))

# unary-exp
cpdef Expression tanh(Expression x): return Expression.from_cexpr(c_tanh(x.c()))
cpdef Expression exp(Expression x): return Expression.from_cexpr(c_exp(x.c()))
cpdef Expression log(Expression x): return Expression.from_cexpr(c_log(x.c()))
cpdef Expression logistic(Expression x): return Expression.from_cexpr(c_logistic(x.c()))
cpdef Expression rectify(Expression x): return Expression.from_cexpr(c_rectify(x.c()))
cpdef Expression log_softmax(Expression x, list restrict=None): 
    if restrict is None:
        return Expression.from_cexpr(c_log_softmax(x.c()))
    cdef vector[unsigned] vec = restrict
    return Expression.from_cexpr(c_log_softmax(x.c(), vec))
cpdef Expression softmax(Expression x): return Expression.from_cexpr(c_softmax(x.c()))
cpdef Expression softsign(Expression x): return Expression.from_cexpr(c_softsign(x.c()))
cpdef Expression transpose(Expression x): return Expression.from_cexpr(c_transpose(x.c()))
cpdef Expression sum_cols(Expression x): return Expression.from_cexpr(c_sum_cols(x.c()))
#expr-opt
cpdef Expression fold_rows(Expression x, unsigned nrows=2): return Expression.from_cexpr(c_fold_rows(x.c(),nrows))
#expr-expr-opt
cpdef Expression pairwise_rank_loss(Expression x, Expression y, float m=1.0): return Expression.from_cexpr(c_pairwise_rank_loss(x.c(), y.c(), m))
cpdef Expression huber_distance(Expression x, Expression y, float c=1.345): return Expression.from_cexpr(c_huber_distance(x.c(), y.c(), c))
#expr-unsigned
cpdef Expression kmax_pooling(Expression x, unsigned k): return Expression.from_cexpr(c_kmax_pooling(x.c(), k))
cpdef Expression pickneglogsoftmax(Expression x, unsigned v): return Expression.from_cexpr(c_pickneglogsoftmax(x.c(), v))

cpdef Expression kmh_ngram(Expression x, unsigned v): return Expression.from_cexpr(c_kmh_ngram(x.c(), v))
cpdef Expression pickrange(Expression x, unsigned v, unsigned u): return Expression.from_cexpr(c_pickrange(x.c(), v, u))
#expr-float
cpdef Expression noise(Expression x, float stddev): return Expression.from_cexpr(c_noise(x.c(), stddev))
cpdef Expression dropout(Expression x, float p): return Expression.from_cexpr(c_dropout(x.c(), p))
#expr-dim
cpdef Expression reshape(Expression x, tuple d): return Expression.from_cexpr(c_reshape(x.c(),Dim(d)))

cpdef Expression esum(list xs):
    cdef vector[CExpression] cvec
    cvec = vector[CExpression]()
    cdef Expression x
    for x in xs: cvec.push_back(x.c())
    #print >> sys.stderr, cvec.size()
    return Expression.from_cexpr(c_sum(cvec))

cpdef Expression average(list xs):
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs: 
        cvec.push_back(x.c())
        print >> sys.stderr,"pushing",cvec.back().i
    return Expression.from_cexpr(c_average(cvec))

cpdef Expression concatenate_cols(list xs):
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs: cvec.push_back(x.c())
    return Expression.from_cexpr(c_concat_cols(cvec))

cpdef Expression concatenate(list xs):
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs: cvec.push_back(x.c())
    return Expression.from_cexpr(c_concat(cvec))


cpdef Expression affine_transform(list exprs):
    cdef Expression e
    cdef vector[CExpression] ves
    for e in exprs:
        ves.push_back(e.c())
    return Expression.from_cexpr(c_affine_transform(ves))

#expr-ptr
cdef class hinge(Expression):
    cdef unsigned index
    def __cinit__(self, Expression x, unsigned index, float m=1.0):
        self.index = index
        self.cg = x.cg
        cdef CExpression e
        e = c_hinge(x.c(), &(self.index), m)
        self.vindex = e.i
    def set_index(self, unsigned i):
        self.index = i

# }}}
    
# {{{ RNNS / Builders
# TODO: unify these with inheritance
cdef class SimpleRNNBuilder:
    cdef CSimpleRNNBuilder *thisptr
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        self.thisptr = new CSimpleRNNBuilder(layers, input_dim, hidden_dim, model.thisptr)
    def __dealloc__(self):
        del self.thisptr
    cpdef new_graph(self, ComputationGraph cg): self.thisptr.new_graph(cg.thisptr[0])
    cpdef start_new_sequence(self, es=None):
        cdef vector[CExpression] ces = vector[CExpression]()
        cdef Expression e
        if es:
            for e in es: ces.push_back(e.c())
        self.thisptr.start_new_sequence(ces)
    cpdef Expression add_input(self, Expression e):
        return Expression.from_cexpr(self.thisptr.add_input(e.c()))
    cpdef rewind_one_step(self): self.thisptr.rewind_one_step()
    cpdef Expression back(self):
        return Expression.from_cexpr(self.thisptr.back())
    cpdef final_h(self):
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.final_h()
        for cexp in cexps:
            res.append(Expression.from_cexpr(cexp))
        return res
    cpdef final_s(self):
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.final_s()
        for cexp in cexps:
            res.append(Expression.from_cexpr(cexp))
        return res
    
cdef class LSTMBuilder:
    cdef CLSTMBuilder *thisptr
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        self.thisptr = new CLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr)
    def __dealloc__(self):
        del self.thisptr
    cpdef new_graph(self, ComputationGraph cg): self.thisptr.new_graph(cg.thisptr[0])
    cpdef start_new_sequence(self, es=None):
        cdef vector[CExpression] ces = vector[CExpression]()
        cdef Expression e
        if es:
            for e in es: ces.push_back(e.c())
        self.thisptr.start_new_sequence(ces)
    cpdef Expression add_input(self, Expression e):
        return Expression.from_cexpr(self.thisptr.add_input(e.c()))
    cpdef rewind_one_step(self): self.thisptr.rewind_one_step()
    cpdef Expression back(self):
        return Expression.from_cexpr(self.thisptr.back())
    cpdef final_h(self):
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.final_h()
        for cexp in cexps:
            res.append(Expression.from_cexpr(cexp))
        return res
    cpdef final_s(self):
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.final_s()
        for cexp in cexps:
            res.append(Expression.from_cexpr(cexp))
        return res
    
# }}}

# {{{ Training 
cdef class SimpleSGDTrainer:
    cdef CSimpleSGDTrainer *thisptr
    def __cinit__(self, Model m, float lam = 1e-6, float e0 = 0.1):
        self.thisptr = new CSimpleSGDTrainer(m.thisptr, lam, e0)
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self, float s):
        self.thisptr.update(s)
    cpdef update_epoch(self, float r = 1.0):
        self.thisptr.update_epoch(r)
    cpdef status(self):
        self.thisptr.status()
#}}}





