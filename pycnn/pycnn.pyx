from pycnn cimport *
cimport pycnn

cdef init():
    cdef char** argv = []
    cdef int argc = 0
    pycnn.Initialize(argc,argv)
init()

#cdef class Dim:
#    cdef CDim *thisptr
#    def __cinit__(self, int n):
#        self.thisptr = new CDim(n)
#    def __dealloc__(self):
#        del self.thisptr
#    def bla(self):
#        return self.thisptr.size()

cdef CDim Dim(dim):
    if isinstance(dim, tuple):
        if len(dim) == 1: return CDim(dim[0])
        if len(dim) == 2: return CDim(dim[0],dim[1])
    return CDim(dim)

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
        if isinstance(dim, tuple):
            if len(dim) == 1:
                p = self.thisptr.add_parameters(CDim(dim[0]))
            elif len(dim) == 2:
                p = self.thisptr.add_parameters(CDim(dim[0],dim[1]))
            else:
                raise "Unsupported dimension",dim
        else: # dim is a single numner
            p = self.thisptr.add_parameters(CDim(dim))
        cdef Parameters pp = Parameters.wrap_ptr(p)
        self.named_params[name] = pp
        return pp

    def __getitem__(self, name):
        return self.named_params[name]

    def __contains__(self, name):
        return name in self.named_params

cdef _add(Expr2 a, Expr2 b): return Expr2.from_cexpr(c_op_add(a.c(), b.c()))
cdef _mul(Expr2 a, Expr2 b): return Expr2.from_cexpr(c_op_mul(a.c(), b.c()))
cdef _neg(Expr2 a): return Expr2.from_cexpr(c_op_neg(a.c()))
cdef _scalarsub(float a, Expr2 b): return Expr2.from_cexpr(c_op_scalar_sub(a, b.c()))
cdef _cmul(Expr2 a, float b): return Expr2.from_cexpr(c_op_scalar_mul(a.c(), b))

cdef class Expr2:
    cdef CComputationGraph *cg
    cdef VariableIndex vindex
    def __cinit__(self):
        self.cg = NULL
        self.vindex = 0
    @staticmethod
    cdef Expr2 from_cexpr(CExpression cexpr):
        self = Expr2()
        self.cg = cexpr.pg
        self.vindex = cexpr.i
        return self
    cdef CExpression c(self):
        return CExpression(self.cg, self.vindex)

    #cdef scalar(self):
    #    return c_as_scalar(self.cg.get_value(self.vindex))

    def __add__(self, other): return _add(self,other)
    def __mul__(self, other):
        if isinstance(self, Expr2) and isinstance(other, Expr2):
            return _mul(self,other)
        elif isinstance(self, (int,float)) or isinstance(other, (int,float)):
            return _cmul(self, other)
        else: raise NotImplementedError()
    def __neg__(self):        return _neg(self)
    def __sub__(self, other):
        if isinstance(self,Expr2) and isinstance(other,Expr2):
            return self+(-other)
        elif isinstance(self,(int,float)) and isinstance(other,Expr2):
            return _scalarsub(self, other)
        elif isinstance(self,Expr2) and isinstance(other,(int, float)):
            return _neg(_scalarsub(other, self))
        else: raise NotImplementedError()


cpdef Expr2 parameter(ComputationGraph g, Parameters p):
    return Expr2.from_cexpr(c_parameter(g.thisptr[0], p.thisptr))

cdef class InputExpr2(Expr2):
    cdef float val
    def __cinit__(self, ComputationGraph g, float s):
        self.val = s
        self.cg = g.thisptr
        cdef CExpression e
        e = c_input(self.cg[0], &(self.val))
        self.vindex = e.i
    def set_input(self, float s):
        self.val = s

cdef class VecInputExpr2(Expr2):
    cdef vector[float] data
    def __cinit__(self, ComputationGraph g, vector[float] data):
        dim = len(data)
        self.data = data
        self.cg = g.thisptr
        cdef CExpression e
        e = c_input(self.cg[0], Dim(dim), &(self.data))
        self.vindex = e.i
    def set_input(self, vector[float] data):
        self.data = data

cdef class LookupExpr2(Expr2):
    cdef unsigned index
    def __cinit__(self, ComputationGraph g, LookupParameters p, unsigned index):
        self.index = index
        self.cg = g.thisptr
        cdef CExpression e
        e = c_lookup(self.cg[0], p.thisptr, &(self.index))
        self.vindex = e.i
    def set_index(self, unsigned index):
        self.index = index

cdef class ConstLookupExpr2(Expr2):
    cdef unsigned index
    def __cinit__(self, ComputationGraph g, LookupParameters p, unsigned index):
        self.index = index
        self.cg = g.thisptr
        cdef CExpression e
        e = c_const_lookup(self.cg[0], p.thisptr, &(self.index))
        self.vindex = e.i
    def set_index(self, unsigned index):
        self.index = index

# binary-exp
cpdef Expr2 cdiv(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_cdiv(x.c(), y.c()))
cpdef Expr2 colwise_add(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_colwise_add(x.c(), y.c()))

cpdef Expr2 cwise_multiply(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_cwise_multiply(x.c(), y.c()))
cpdef Expr2 dot_product(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_dot_product(x.c(), y.c()))
cpdef Expr2 squared_distance(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_squared_distance(x.c(), y.c()))
cpdef Expr2 l1_distance(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_l1_distance(x.c(), y.c()))
cpdef Expr2 binary_log_loss(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_binary_log_loss(x.c(), y.c()))
cpdef Expr2 conv1d_narrow(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_conv1d_narrow(x.c(), y.c()))
cpdef Expr2 conv1d_wide(Expr2 x, Expr2 y): return Expr2.from_cexpr(c_conv1d_wide(x.c(), y.c()))

# unary-exp
cpdef Expr2 tanh(Expr2 x): return Expr2.from_cexpr(c_tanh(x.c()))
cpdef Expr2 exp(Expr2 x): return Expr2.from_cexpr(c_exp(x.c()))
cpdef Expr2 log(Expr2 x): return Expr2.from_cexpr(c_log(x.c()))
cpdef Expr2 logistic(Expr2 x): return Expr2.from_cexpr(c_logistic(x.c()))
cpdef Expr2 rectify(Expr2 x): return Expr2.from_cexpr(c_rectify(x.c()))
cpdef Expr2 log_softmax(Expr2 x, list restrict=None): 
    if restrict is None:
        return Expr2.from_cexpr(c_log_softmax(x.c()))
    cdef vector[unsigned] vec = restrict
    return Expr2.from_cexpr(c_log_softmax(x.c(), vec))
cpdef Expr2 softmax(Expr2 x): return Expr2.from_cexpr(c_softmax(x.c()))
cpdef Expr2 softsign(Expr2 x): return Expr2.from_cexpr(c_softsign(x.c()))
cpdef Expr2 transpose(Expr2 x): return Expr2.from_cexpr(c_transpose(x.c()))
cpdef Expr2 sum_cols(Expr2 x): return Expr2.from_cexpr(c_sum_cols(x.c()))
#expr-opt
cpdef Expr2 fold_rows(Expr2 x, unsigned nrows=2): return Expr2.from_cexpr(c_fold_rows(x.c(),nrows))
#expr-expr-opt
cpdef Expr2 pairwise_rank_loss(Expr2 x, Expr2 y, float m=1.0): return Expr2.from_cexpr(c_pairwise_rank_loss(x.c(), y.c(), m))
cpdef Expr2 huber_distance(Expr2 x, Expr2 y, float c=1.345): return Expr2.from_cexpr(c_huber_distance(x.c(), y.c(), c))
#expr-unsigned
cpdef Expr2 kmax_pooling(Expr2 x, unsigned k): return Expr2.from_cexpr(c_kmax_pooling(x.c(), k))
cpdef Expr2 pickneglogsoftmax(Expr2 x, unsigned v): return Expr2.from_cexpr(c_pickneglogsoftmax(x.c(), v))
cpdef Expr2 kmh_ngram(Expr2 x, unsigned v): return Expr2.from_cexpr(c_kmh_ngram(x.c(), v))
cpdef Expr2 pickrange(Expr2 x, unsigned v, unsigned u): return Expr2.from_cexpr(c_pickrange(x.c(), v, u))
#expr-float
cpdef Expr2 noise(Expr2 x, float stddev): return Expr2.from_cexpr(c_noise(x.c(), stddev))
cpdef Expr2 dropout(Expr2 x, float p): return Expr2.from_cexpr(c_dropout(x.c(), p))
#expr-dim
cpdef Expr2 reshape(Expr2 x, tuple d): return Expr2.from_cexpr(c_reshape(x.c(),Dim(d)))
#expr-ptr
cdef class hinge(Expr2):
    cdef unsigned index
    def __cinit__(self, Expr2 x, unsigned index, float m=1.0):
        self.index = index
        self.cg = x.cg
        cdef CExpression e
        e = c_hinge(x.c(), &(self.index), m)
        self.vindex = e.i
    def set_index(self, unsigned i):
        self.index = i

cdef class pick(Expr2):
    cdef unsigned index
    def __cinit__(self, Expr2 x, unsigned index):
        self.index = index
        self.cg = x.cg
        cdef CExpression e
        e = c_pick(x.c(), &(self.index))
        self.vindex = e.i
    def set_index(self, unsigned i):
        self.index = i

    
cdef class ComputationGraph:
    cdef CComputationGraph *thisptr, 
    def __cinit__(self):
        self.thisptr = new CComputationGraph()
    def __dealloc__(self):
        del self.thisptr

    cpdef renew(self):
        del self.thisptr
        self.thisptr = new CComputationGraph()
        return self

    def parameters(self, model, name, dim=None):
        cdef Parameters params
        cdef Expr2 result
        if name in model:
            params = model[name]
        else:
            assert(dim is not None)
            params = model.add_parameters(name, dim)
        result = Expr2.from_cexpr(c_parameter(self.thisptr[0], params.thisptr))
        #result = self._expr(parameter(self.thisptr[0], params.thisptr))
        #result = Expression.wrap_ptr(&(parameter(self.thisptr[0], params.thisptr)))
        return result

    cpdef forward_scalar(self):
        return c_as_scalar(self.thisptr.forward())

    cpdef forward_vec(self):
        return c_as_vector(self.thisptr.forward())

    cpdef backward(self):
        self.thisptr.backward()


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

    

