# on numpy arrays, see: https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
from __future__ import print_function
import sys
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
import numpy as np

# python3 pickle already uses the c implementaion 
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
import os.path
# TODO:
#  - set random seed (in DYNET)
#  - better input / output support
#    WORKS, but need to be unified? for example, why "pick" takes a pointer to int, and "squared_distance" takes an expression?
#  - load embeddings file
#  - load/save models
#  - NOTE: why do we need to filter short sentences in rnnlm.py or crash??

# TODO:
#  c2w.h   (build a word-from-letters encoder)
#  dict.h  -- do we even need it?

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

from _dynet cimport *
cimport _dynet as dynet


cdef init(random_seed=None):
    cdef int argc = len(sys.argv)
    cdef char** c_argv
    args = [bytearray(x, encoding="utf-8") for x in sys.argv]
    c_argv = <char**>malloc(sizeof(char*) * len(args)) # TODO check failure?
    for idx, s in enumerate(args):
        c_argv[idx] = s

    if random_seed is None:
        dynet.initialize(argc,c_argv, 0)
    else:
        if random_seed == 0: random_seed = 1
        dynet.initialize(argc,c_argv, random_seed)
    free(c_argv)

init() # TODO: allow different random seeds

cdef CDim Dim(dim, unsigned int batch_size=1):
    """
    dim: either a tuple or an int
    """
    cdef vector[long] cvec
    if isinstance(dim, tuple):
        for d in dim: cvec.push_back(d)
    elif isinstance(dim, (int, float)):
        cvec.push_back(dim)
    else:
        raise "Unsupported dimension",dim

    if batch_size > 1:
        return CDim(cvec, batch_size)
    else:
        return CDim(cvec)

cdef c_tensor_as_np(CTensor &t):
    # TODO: make more efficient, with less copy
    arr = np.array(c_as_vector(t))
    if t.d.ndims() == 1: return arr
    else: return arr.reshape(t.d.rows(), t.d.cols(),order='F')

# {{{ Model / Parameters 
cdef class Parameters:
    cdef CParameters thisptr # TODO -- no longer pointer
    cdef int _version
    cdef Expression _expr
    def __cinit__(self):
        #self.thisptr = p
        self._version = -1
        pass
    @staticmethod
    cdef wrap_ptr(CParameters ptr):
        self = Parameters()
        self.thisptr = ptr
        return self

    cpdef shape(self):
        if self.thisptr.get().dim.ndims() == 1: return (self.thisptr.get().dim.rows())
        return (self.thisptr.get().dim.rows(), self.thisptr.get().dim.cols())

    cpdef as_array(self):
        """
        Return as a numpy array.
        """
        cdef CTensor t
        return c_tensor_as_np(self.thisptr.get().values)

    # TODO: make more efficient
    cpdef load_array(self, arr):
        assert(False),"This method is depracated. Use instead model.parameters_from_numpy(arr)."
        cdef CTensor t
        cdef float* vals
        t = self.thisptr.get().values
        shape = arr.shape
        if len(shape) == 1:
            assert(t.d.ndims() == 1)
            assert(t.d.size() == arr.size)
        if len(shape) == 2:
            assert(t.d.rows() == shape[0] and t.d.cols() == shape[1])
        vals = t.v
        arr = arr.flatten()
        for i in xrange(arr.size):
            vals[i] = arr[i]

    cpdef zero(self): self.thisptr.zero()

    cpdef bool is_updated(self): return self.thisptr.is_updated()
    cpdef set_updated(self, bool b): self.thisptr.set_updated(b)

    cpdef Expression expr(self):
        if cg_version() != self._version:
            self._version = cg_version()
            self._expr = Expression.from_cexpr(_cg.version(), c_parameter(_cg.thisptr[0], self.thisptr))
        return self._expr



cdef class LookupParameters:
    cdef CLookupParameters thisptr # TODO -- no longer pointer
    def __cinit__(self):
        pass
    @staticmethod
    cdef wrap_ptr(CLookupParameters ptr):
        self = LookupParameters()
        self.thisptr = ptr
        return self

    cpdef init_from_array(self, arr):
        if len(arr) > self.thisptr.get().values.size():
            raise Exception("too many rows")
        if arr.shape[1] != self.thisptr.get().values[0].d.rows():
            raise Exception("dim mismatch")
        cdef vector[float] r
        for i,row in enumerate(arr):
            self.init_row(i, row)

    cpdef shape(self):
        if self.thisptr.get().dim.cols() != 1:
            return (self.thisptr.get().values.size(), self.thisptr.get().dim.rows(), self.thisptr.get().dim.cols())
        return (self.thisptr.get().values.size(), self.thisptr.get().dim.rows())

    def __getitem__(self, int i):
        return lookup(self, i)

    cpdef batch(self, vector[unsigned] i):
        return lookup_batch(self, i)

    cpdef init_row(self, unsigned i, vector[float] row):
        self.thisptr.initialize(i, row)

    cpdef as_array(self):
        """
        Return as a numpy array.
        """
        cdef vector[CTensor] vals
        vals = self.thisptr.get().values
        return np.vstack([c_tensor_as_np(t).reshape(1,-1,order='F') for t in vals])

    cpdef zero(self): self.thisptr.zero()

    cpdef bool is_updated(self): return self.thisptr.is_updated()
    cpdef set_updated(self, bool b): self.thisptr.set_updated(b)

# TODO document this
class Saveable(object):
    def __init__(self):
        pass

    def __getstate__(self):
        odict = dict()
        params = self.get_components()
        for k,v in self.__dict__.items(): # remove unpicklable things which we save otherwise
            if v not in params:
                odict[k] = v
        return odict

    def get_components(self):
        """
        List of parameter-containing components that are
        members of this object and are created by it.
        """
        return NotImplemented

    def restore_components(self, components):
        return NotImplemented


# Initializers
cdef class PyInitializer:
    cdef CParameterInit *initializer
    def __init__(self):
        assert(False),"Do not create PyInitializer directly."
    def __dealloc__(self):
        del self.initializer

cdef class NormalInitializer(PyInitializer):
    def __init__(self, float mean=0, var=1):
        self.initializer = new CParameterInitNormal(mean, var)

cdef class UniformInitializer(PyInitializer):
    def __init__(self, float scale):
        self.initializer = new CParameterInitUniform(scale)

cdef class ConstInitializer(PyInitializer):
    def __init__(self, float c):
        self.initializer = new CParameterInitConst(c)

cdef class IdentityInitializer(PyInitializer):
    def __init__(self):
        self.initializer = new CParameterInitIdentity()

cdef class GlorotInitializer(PyInitializer):
    def __init__(self, bool is_lookup=False):
        self.initializer = new CParameterInitGlorot(is_lookup)

#cdef class SaxeInitializer(PyInitializer):
#    def __init__(self):
#        self.initializer = new CParameterInitSaxe()

cdef class FromFileInitializer(PyInitializer):
    def __init__(self, string fname):
        self.initializer = new CParameterInitFromFile(fname)

cdef class NumpyInitializer(PyInitializer):
    def __init__(self, array):
        self.initializer = new CParameterInitFromVector(self.vec_from_array(array))

    cdef vector[float] vec_from_array(self, arr): # TODO make efficient
        cdef vector[float] vals
        shape = arr.shape
        arr = arr.flatten(order='F')
        for i in xrange(arr.size):
            vals.push_back(arr[i])
        return vals


cdef class Model: # {{{
    cdef CModel *thisptr
    def __cinit__(self):
        self.thisptr = new CModel()
    def __init__(self):
        pass

    def __dealloc__(self): del self.thisptr

    @staticmethod
    def from_file(fname):
        model = Model()
        res = model.load(fname)
        return model, res

    # TODO: for debug, remove
    cpdef pl(self): return self.thisptr.parameters_list().size()

    cpdef parameters_from_numpy(self, array):
        dim = array.shape
        cdef CParameters p = self.thisptr.add_parameters(Dim(dim), deref(NumpyInitializer(array).initializer))
        cdef Parameters pp = Parameters.wrap_ptr(p)
        return pp

    cpdef add_parameters(self, dim, PyInitializer init=None):
        assert(isinstance(dim,(tuple,int)))
        cdef CParameters p
        cdef CParameterInit *initializer
        if init is None:
            init = GlorotInitializer()
        initializer = init.initializer
        p = self.thisptr.add_parameters(Dim(dim), deref(initializer))
        cdef Parameters pp = Parameters.wrap_ptr(p)
        return pp

    cpdef add_lookup_parameters(self, dim, PyInitializer init=None):
        assert(isinstance(dim, tuple))
        cdef int nids = dim[0]
        rest = tuple(dim[1:])
        if init is None:
            init = GlorotInitializer(True)
        initializer = init.initializer
        cdef CLookupParameters p = self.thisptr.add_lookup_parameters(nids, Dim(rest), deref(initializer))
        cdef LookupParameters pp = LookupParameters.wrap_ptr(p)
        return pp

    def save_all(self, string fname):
        save_dynet_model(fname, self.thisptr)

    cdef load_all(self, string fname):
        load_dynet_model(fname, self.thisptr)

    cdef _save_one(self, component, CModelSaver *saver, fh, pfh):
        # would be nicer to have polymorphism/dispatch-by-type
        # but we cannot because we need to bind to the c-type.
        c = component
        if isinstance(c, Parameters):
            fh.write("param ")
            saver.add_parameter((<Parameters>c).thisptr)
        elif isinstance(c, LookupParameters):
            fh.write("lookup ")
            saver.add_lookup_parameter((<LookupParameters>c).thisptr)
        elif isinstance(c, GRUBuilder):
            fh.write("gru_builder ")
            saver.add_gru_builder((<CGRUBuilder*>(<GRUBuilder>c).thisptr)[0])
        elif isinstance(c, LSTMBuilder):
            fh.write("lstm_builder ")
            saver.add_lstm_builder((<CLSTMBuilder*>(<LSTMBuilder>c).thisptr)[0])
        elif isinstance(c, VanillaLSTMBuilder):
            fh.write("vanilla_lstm_builder ")
            saver.add_vanilla_lstm_builder((<CVanillaLSTMBuilder*>(<VanillaLSTMBuilder>c).thisptr)[0])
        elif isinstance(c, SimpleRNNBuilder):
            saver.add_srnn_builder((<CSimpleRNNBuilder*>(<SimpleRNNBuilder>c).thisptr)[0])
            fh.write("srnn_builder ")
        elif isinstance(c, BiRNNBuilder):
            fh.write("birnn_builder~%d " % (2 * len(c.builder_layers)))
            for (f,b) in c.builder_layers:
                self._save_one(f,saver,fh,pfh)
                self._save_one(b,saver,fh,pfh)
        elif isinstance(c, Saveable):
            cs = c.get_components()
            fh.write("user~%d " % len(cs))
            pickle.dump(c,pfh)
            for subc in cs:
                self._save_one(subc,saver,fh,pfh)
        else:
            raise TypeError("Cannot save model component of type %s" % type(c))

    def save(self, fname, components=None):
        if not components:
            self.save_all(fname.encode())
            return
        fh = open(fname+".pym","w")
        pfh = open(fname+".pyk","wb")
        cdef CModelSaver *saver = new CModelSaver(fname.encode(), self.thisptr)
        for c in components:
            self._save_one(c,saver,fh,pfh)
        saver.done()
        fh.close()
        pfh.close()
        del saver

    cdef _load_one(self, itypes, CModelLoader *loader, pfh):
        cdef CParameters p
        cdef CLookupParameters lp
        cdef GRUBuilder gb_
        cdef LSTMBuilder lb_
        cdef VanillaLSTMBuilder vlb_
        cdef SimpleRNNBuilder sb_
        tp = next(itypes)
        if tp == "param":
            loader.fill_parameter(p)
            param = Parameters.wrap_ptr(p)
            return param
        elif tp == "lookup":
            loader.fill_lookup_parameter(lp)
            param = LookupParameters.wrap_ptr(lp)
            return param
        elif tp == "gru_builder":
            gb_ = GRUBuilder(0,0,0,self) # empty builder
            loader.fill_gru_builder((<CGRUBuilder *>gb_.thisptr)[0])
            return gb_
        elif tp == "lstm_builder":
            lb_ = LSTMBuilder(0,0,0,self) # empty builder
            loader.fill_lstm_builder((<CLSTMBuilder *>lb_.thisptr)[0])
            return lb_
        elif tp == "vanilla_lstm_builder":
            vlb_ = VanillaLSTMBuilder(0,0,0,self) # empty builder
            loader.fill_vanilla_lstm_builder((<CVanillaLSTMBuilder *>vlb_.thisptr)[0])
            return vlb_
        elif tp == "srnn_builder":
            sb_ = SimpleRNNBuilder(0,0,0,self) # empty builder
            loader.fill_srnn_builder((<CSimpleRNNBuilder *>sb_.thisptr)[0])
            return sb_
        elif tp.startswith("birnn_builder~"):
            tp,num = tp.split("~",1)
            num = int(num)
            items = [self._load_one(itypes, loader, pfh) for _ in xrange(num)]
            return BiRNNBuilder(None, None, None, None, None, list(zip(items[0::2], items[1::2])))
        elif tp.startswith("user~"):
            # user defiend type
            tp,num = tp.split("~",1)
            saveable = pickle.load(pfh)
            num = int(num)
            items = [self._load_one(itypes, loader, pfh) for _ in xrange(num)]
            saveable.restore_components(items)
            return saveable
        else:
            print("Huh?")
            assert False,"unsupported type " + tp

    cpdef load(self, fname):
        if not os.path.isfile(fname+".pym"):
            self.load_all(fname.encode())
            return
        with open(fname+".pym","r") as fh:
            types = fh.read().strip().split()

        cdef CModelLoader *loader = new CModelLoader(fname.encode(), self.thisptr)
        with open(fname+".pyk","rb") as pfh:
            params = []
            itypes = iter(types)
            while True: # until iterator is done
                try:
                    param = self._load_one(itypes,loader,pfh)
                except StopIteration: break
                params.append(param)
        loader.done()
        del loader
        return params
    #}}}

# }}}

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

cdef class UnsignedVectorValue:
    cdef vector[unsigned] *vals
    def __cinit__(self, vals):
        self.vals = new vector[unsigned]()
        self.set(vals)
    def __dealloc__(self):
        del self.vals
    def set(self, newval):
        self.vals.clear()
        for f in newval: self.vals.push_back(f)
    def get(self): return deref(self.vals)
    def size(self): return len(deref(self.vals))
    cdef vector[unsigned]* addr(self): return self.vals

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

cdef int SECRET = 923148
cdef ComputationGraph _cg = ComputationGraph(SECRET)

def cg_version(): return _cg._cg_version
def renew_cg(): return _cg.renew()
def print_text_graphviz(): return _cg.print_graphviz()
def cg_checkpoint(): _cg.checkpoint()
def cg_revert():     _cg.revert()

cpdef ComputationGraph cg():
    global _cg
    return _cg

cdef class ComputationGraph:
    cdef CComputationGraph *thisptr, 
    cdef list _inputs
    cdef int _cg_version
    def __cinit__(self, int guard=0):
        if guard != SECRET: raise RuntimeError("Do not instantiate ComputationGraph directly. Use dynet.renew_cg()")
        self.thisptr = new CComputationGraph()
        self._inputs = []
        self._cg_version = 0
    def __dealloc__(self):
        del self.thisptr

    cpdef renew(self):
        del self.thisptr
        self.thisptr = new CComputationGraph()
        self._inputs = []
        self._cg_version += 1
        return self

    cpdef version(self): return self._cg_version

    def parameters(self, Parameters params):
        cdef Expression result
        result = Expression.from_cexpr(self._cg_version, c_parameter(self.thisptr[0], params.thisptr))
        return result

    #def params_from_model(self, model):
    #    results = {}
    #    for name in model.regular_parameters():
    #        results[name] = self.parameters(model[name])
    #    for name in model.lookup_parameters():
    #        results[name] = self.lookup(model[name])
    #    return results

    cpdef forward_scalar(self, VariableIndex index):
        return c_as_scalar(self.thisptr.forward(index))

    cpdef inc_forward_scalar(self, VariableIndex index):
        return c_as_scalar(self.thisptr.incremental_forward(index))

    cpdef forward_vec(self, VariableIndex index):
        return c_as_vector(self.thisptr.forward(index))

    cpdef inc_forward_vec(self, VariableIndex index):
        return c_as_vector(self.thisptr.incremental_forward(index))

    cpdef forward(self, VariableIndex index): self.thisptr.forward(index)
    cpdef inc_forward(self, VariableIndex index): self.thisptr.incremental_forward(index)

    cpdef backward(self, VariableIndex index):
        self.thisptr.backward(index)

    cpdef print_graphviz(self):
        self.thisptr.print_graphviz()

    cpdef void checkpoint(self):
        self.thisptr.checkpoint()

    cpdef void revert(self):
        self.thisptr.revert()

    # DYNET handles changing inputs keeping pointers to memoty locations.
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
    cdef inputValue(self, float v = 0.0):
        return _inputExpression(self, v)
    cdef inputVector(self, int dim):
        return _vecInputExpression(self, vector[float](dim))
    cdef inputVectorLiteral(self, vector[float] v):
        return _vecInputExpression(self, v)
    cdef inputMatrix(self, int d1, int d2):
        return _vecInputExpression(self, vector[float](d1*d2), (d1,d2))
    def inputMatrixLiteral(self, vector[float] v, tuple d):
        return _vecInputExpression(self, v, d)
    cdef lookup(self, LookupParameters p, unsigned v = 0, update=True):
        return _lookupExpression(self, p, v, update)
    cdef lookup_batch(self, LookupParameters p, vector[unsigned] vs, update=True):
        return _lookupBatchExpression(self, p, vs, update)
    cdef outputPicker(self, Expression e, unsigned v=0):
        r = _pickerExpression(self, e, v)
        return r
    cdef outputBatchPicker(self, Expression e, vector[unsigned] vs):
        r = _pickerBatchExpression(self, e, vs)
        return r



# }}}

#{{{ Expressions
cdef ensure_freshness(Expression a):
    if a.cg_version != _cg.version(): raise ValueError("Attempt to use a stale expression.")

cdef _add(Expression a, Expression b): ensure_freshness(b); return Expression.from_cexpr(a.cg_version, c_op_add(a.c(), b.c()))
cdef _mul(Expression a, Expression b): ensure_freshness(b); return Expression.from_cexpr(a.cg_version, c_op_mul(a.c(), b.c()))
cdef _neg(Expression a): return Expression.from_cexpr(a.cg_version, c_op_neg(a.c()))
cdef _scalarsub(float a, Expression b): ensure_freshness(b); return Expression.from_cexpr(b.cg_version, c_op_scalar_sub(a, b.c()))
cdef _cadd(Expression a, float b): return Expression.from_cexpr(a.cg_version, c_op_scalar_add(a.c(), b))
cdef _cmul(Expression a, float b): return Expression.from_cexpr(a.cg_version, c_op_scalar_mul(a.c(), b))
cdef _cdiv(Expression a, float b): return Expression.from_cexpr(a.cg_version, c_op_scalar_div(a.c(), b))

cdef class Expression: #{{{
    #cdef CComputationGraph* cg
    # cg is a singleton, so there is no need to keep it inside the expression.
    # not keeping cg() in the expression will preserve memory.
    # if DYNET comes to support multiple computation graphs, this will need to change.
    cdef inline ComputationGraph cg(self):
        return cg()
    cdef inline CComputationGraph* cgp(self):
        return cg().thisptr

    cdef VariableIndex vindex
    cdef int cg_version
    def __cinit__(self):
        #self.cg = NULL
        self.vindex = 0
    @staticmethod
    cdef Expression from_cexpr(int cgv, CExpression cexpr):
        if cgv != _cg._cg_version: raise ValueError("Attempt to use a stale expression, from a previous Computation Graph.")
        self = Expression()
        #self.cg = cexpr.pg
        self.vindex = cexpr.i
        self.cg_version = cgv
        return self
    cdef CExpression c(self):
        return CExpression(self.cgp(), self.vindex)

    cpdef dim(self):
        cdef CDim d;
        d=self.c().dim()
        return (d.size(), d.rows(), d.cols(), d.batch_elems())

    def __repr__(self):
        return str(self)
    def __str__(self):
        return "expression %s/%s" % (<int>self.vindex, self.cg_version)

    # __getitem__ and __getslice__ in one for python 3 compatibility
    def __getitem__(self, index):
        assert isinstance(index, (int, slice))
        cdef int rows = self.c().dim().rows()
        cdef int i, j
        if isinstance(index, int):
            i = index
            if i > rows - 1:
                raise IndexError("Index too large: %d > %d" % (i, rows - 1))
            if i < -rows:
                raise IndexError("Index too small: %d < %d" % (i, -rows))
            if i < 0:
                i += rows
            return pick(self, i)
        else:
            i = 0
            j = rows
            if index.start is not None:
                i = index.start
                if i > rows - 1:
                    raise IndexError("Start index too large: %d > %d" % (i, rows - 1))
                if i < -rows:
                    raise IndexError("Start index too small: %d < %d" % (i, -rows))
                if i < 0:
                    i += rows
            if index.stop is not None:
                j = index.stop
                if j > rows:
                    raise IndexError("Stop index too large: %d > %d" % (j, rows))
                if j < -rows + 1:
                    raise IndexError("Stop index too small: %d < %d" % (j, -rows + 1))
                if j < 0:
                    j += rows
            if i >= j:
                raise ValueError("Improper slice: start index must come strictly before stop index")
            if index.step is not None:
                raise ValueError("Step sizes not yet supported.")
            return pickrange(self, i, j)

    cpdef scalar_value(self, recalculate=False):
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        if recalculate: self.cg().forward(self.vindex) # TODO: make recalculate run on the entire graph, not only up to here?
        return c_as_scalar(self.cgp().get_value(self.vindex))

    cpdef vec_value(self, recalculate=False):
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        if recalculate: self.cg().forward(self.vindex)
        return c_as_vector(self.cgp().get_value(self.vindex))

    cpdef npvalue(self, recalculate=False):
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        cdef CTensor t
        cdef CDim dim
        if recalculate: self.cg().forward(self.vindex)
        t = self.cgp().get_value(self.vindex)
        dim = t.d
        arr = np.array(c_as_vector(t))
        if dim.batch_elems() > 1:
            if dim.ndims() == 1:
                arr = arr.reshape(dim.rows(), dim.batch_elems(),order='F')
            elif dim.ndims() == 2:
                arr = arr.reshape(dim.rows(), dim.cols(), dim.batch_elems(),order='F')
            else:
                assert(False)
            return arr
        if dim.ndims() == 2:
            arr = arr.reshape(dim.rows(), dim.cols(),order='F')
        return arr

    cpdef value(self, recalculate=False):
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        cdef CTensor t
        if recalculate: self.cg().forward(self.vindex)
        t = self.cgp().get_value(self.vindex)
        if t.d.ndims() == 2:
            return self.npvalue()
        vec = self.vec_value()
        if len(vec) == 1: return vec[0]
        return vec

    # TODO this runs incremental forward on the entire graph, may not be optimal in terms of efficiency.
    cpdef forward(self, recalculate=False):
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        if recalculate: self.cg().forward(self.vindex)
        else: self.cg().inc_forward(self.vindex)

    cpdef backward(self):
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        self.cgp().backward(self.vindex)

    def __add__(self, other):
        if isinstance(self, Expression) and isinstance(other, Expression):
            return _add(self,other)
        elif isinstance(self, (int,float)) or isinstance(other, (int,float)):
            return _cadd(self, other)
        else: raise NotImplementedError()
    def __mul__(self, other):
        if isinstance(self, Expression) and isinstance(other, Expression):
            return _mul(self,other)
        elif isinstance(self, (int,float)) or isinstance(other, (int,float)):
            return _cmul(self, other)
        else: raise NotImplementedError()
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
#}}}

#cdef Expression _parameter(ComputationGraph g, Parameters p):
#    return Expression.from_cexpr(g.version(), c_parameter(g.thisptr[0], p.thisptr))

def parameter(Parameters p): return p.expr()

# {{{ Mutable Expressions
#     These depend values that can be set by the caller

cdef class _inputExpression(Expression):
    cdef FloatValue val
    def __cinit__(self, ComputationGraph g, float s):
        self.val = FloatValue(s)
        #self.cg = g.thisptr
        self.cg_version = g.version()
        cdef CExpression e
        e = c_input(self.cgp()[0], self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self, float s):
        self.cgp().invalidate()
        self.val.set(s)

def scalarInput(float s):
    return _cg.inputValue(s)

cdef class _vecInputExpression(Expression):
    cdef FloatVectorValue val
    def __cinit__(self, ComputationGraph g, vector[float] val, dim=None):
        self.val = FloatVectorValue(val)
        if dim is None: dim = self.val.size()
        #self.cg = g.thisptr
        self.cg_version = g.version()
        cdef CExpression e
        e = c_input(self.cgp()[0], Dim(dim), self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self, vector[float] data):
        self.cgp().invalidate()
        self.val.set(data)

def vecInput(int dim):
    return _cg.inputVector(dim)

def inputVector(vector[float] v):
    return _cg.inputVectorLiteral(v)

def matInput(int d1, int d2):
    return _cg.inputMatrix(d1, d2)

def inputMatrix(vector[float] v, tuple d):
    return _cg.inputMatrixLiteral(v, d)

cdef class _lookupExpression(Expression):
    cdef UnsignedValue val
    def __cinit__(self, ComputationGraph g, LookupParameters p, unsigned index=0, update=True):
        self.val = UnsignedValue(index)
        #self.cg = g.thisptr
        self.cg_version = g.version()
        cdef CExpression e
        if update:
            e = c_lookup(self.cgp()[0], p.thisptr, self.val.addr())
        else:
            e = c_const_lookup(self.cgp()[0], p.thisptr, self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self,i):
        self.cgp().invalidate()
        self.val.set(i)

cdef class _lookupBatchExpression(Expression):
    cdef UnsignedVectorValue val
    def __cinit__(self, ComputationGraph g, LookupParameters p, vector[unsigned] indices, update=True):
        self.val = UnsignedVectorValue(indices)
        self.cg_version = g.version()
        cdef CExpression e
        if update:
            e = c_lookup(self.cgp()[0], p.thisptr, self.val.addr())
        else:
            e = c_const_lookup(self.cgp()[0], p.thisptr, self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self,i):
        self.cgp().invalidate()
        self.val.set(i)

def lookup(LookupParameters p, unsigned index=0, update=True):
    return _cg.lookup(p, index, update)

def lookup_batch(LookupParameters p, vector[unsigned] indices, update=True):
    return _cg.lookup_batch(p, indices, update)

cdef class _pickerExpression(Expression):
    cdef UnsignedValue val
    def __cinit__(self, ComputationGraph g, Expression e, unsigned index=0):
        self.val = UnsignedValue(index)
        #self.cg = e.cg
        self.cg_version = g.version()
        cdef CExpression ce
        ce = c_pick(e.c(), self.val.addr())
        self.vindex = ce.i
        g._inputs.append(self)
    def set_index(self,i):
        self.cgp().invalidate()
        self.val.set(i)

def pick(Expression e, unsigned index=0):
    return _cg.outputPicker(e, index)

cdef class _pickerBatchExpression(Expression):
    cdef UnsignedVectorValue val
    def __cinit__(self, ComputationGraph g, Expression e, vector[unsigned] indices):#
        self.val = UnsignedVectorValue(indices)
        self.cg_version = g.version()
        cdef CExpression ce
        ce = c_pick(e.c(), self.val.addr())
        self.vindex = ce.i
        g._inputs.append(self)
    def set_index(self,i):
        self.cgp().invalidate()
        self.val.set(i)

def pick_batch(Expression e, vector[unsigned] indices):
    return _cg.outputBatchPicker(e, indices)

cdef class _hingeExpression(Expression):
    cdef UnsignedValue val
    def __cinit__(self, ComputationGraph g, Expression x, unsigned index, float m=1.0):
        self.val = UnsignedValue(index)
        #self.cg = x.cg
        self.cg_version = g.version()
        cdef CExpression e
        e = c_hinge(x.c(), self.val.addr(), m)
        self.vindex = e.i
        g._inputs.append(self)
    def set_index(self, unsigned i):
        self.cgp().invalidate()
        self.val.set(i)

def hinge(Expression x, unsigned index, float m=1.0):
    return _hingeExpression(_cg, x, index, m)

# }}}

cpdef Expression zeroes(dim, int batch_size=1): return Expression.from_cexpr(_cg.version(), c_zeroes(_cg.thisptr[0], CDim(dim, batch_size)))
cpdef Expression random_normal(dim, int batch_size=1): return Expression.from_cexpr(_cg.version(), c_random_normal(_cg.thisptr[0], CDim(dim, batch_size)))
cpdef Expression random_bernoulli(dim, float p, float scale=1.0, int batch_size=1): return Expression.from_cexpr(_cg.version(), c_random_bernoulli(_cg.thisptr[0], CDim(dim, batch_size), p, scale))
cpdef Expression random_uniform(dim, float left, float right, int batch_size=1): return Expression.from_cexpr(_cg.version(), c_random_uniform(_cg.thisptr[0], CDim(dim, batch_size), left, right))

cpdef Expression nobackprop(Expression x): return Expression.from_cexpr(x.cg_version, c_nobackprop(x.c()))

# binary-exp
cpdef Expression cdiv(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_cdiv(x.c(), y.c()))
cpdef Expression cmult(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_cmult(x.c(), y.c()))
cpdef Expression colwise_add(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_colwise_add(x.c(), y.c()))

cpdef Expression trace_of_product(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_trace_of_product(x.c(), y.c()))
cpdef Expression dot_product(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_dot_product(x.c(), y.c()))
cpdef Expression squared_norm(Expression x): return Expression.from_cexpr(x.cg_version, c_squared_norm(x.c()))
cpdef Expression squared_distance(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_squared_distance(x.c(), y.c()))
cpdef Expression l1_distance(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_l1_distance(x.c(), y.c()))
cpdef Expression binary_log_loss(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_binary_log_loss(x.c(), y.c()))
cpdef Expression conv1d_narrow(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_conv1d_narrow(x.c(), y.c()))
cpdef Expression conv1d_wide(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_conv1d_wide(x.c(), y.c()))
cpdef Expression filter1d_narrow(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_filter1d_narrow(x.c(), y.c()))

# unary-exp
cpdef Expression tanh(Expression x): return Expression.from_cexpr(x.cg_version, c_tanh(x.c()))
cpdef Expression exp(Expression x): return Expression.from_cexpr(x.cg_version, c_exp(x.c()))
cpdef Expression square(Expression x): return Expression.from_cexpr(x.cg_version, c_square(x.c()))
cpdef Expression sqrt(Expression x): return Expression.from_cexpr(x.cg_version, c_sqrt(x.c()))
cpdef Expression erf(Expression x): return Expression.from_cexpr(x.cg_version, c_erf(x.c()))
cpdef Expression cube(Expression x): return Expression.from_cexpr(x.cg_version, c_cube(x.c()))
cpdef Expression log(Expression x): return Expression.from_cexpr(x.cg_version, c_log(x.c()))
cpdef Expression lgamma(Expression x): return Expression.from_cexpr(x.cg_version, c_lgamma(x.c()))
cpdef Expression logistic(Expression x): return Expression.from_cexpr(x.cg_version, c_logistic(x.c()))
cpdef Expression rectify(Expression x): return Expression.from_cexpr(x.cg_version, c_rectify(x.c()))
cpdef Expression log_softmax(Expression x, list restrict=None): 
    if restrict is None:
        return Expression.from_cexpr(x.cg_version, c_log_softmax(x.c()))
    cdef vector[unsigned] vec = restrict
    return Expression.from_cexpr(x.cg_version, c_log_softmax(x.c(), vec))
cpdef Expression softmax(Expression x): return Expression.from_cexpr(x.cg_version, c_softmax(x.c()))
cpdef Expression sparsemax(Expression x): return Expression.from_cexpr(x.cg_version, c_sparsemax(x.c()))
cpdef Expression softsign(Expression x): return Expression.from_cexpr(x.cg_version, c_softsign(x.c()))
cpdef Expression pow(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_pow(x.c(), y.c()))
cpdef Expression bmin(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_bmin(x.c(), y.c()))
cpdef Expression bmax(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_bmax(x.c(), y.c()))
cpdef Expression transpose(Expression x): return Expression.from_cexpr(x.cg_version, c_transpose(x.c()))
cpdef Expression select_rows(Expression x, vector[unsigned] rs): return Expression.from_cexpr(x.cg_version, c_select_rows(x.c(), rs))
cpdef Expression select_cols(Expression x, vector[unsigned] rs): return Expression.from_cexpr(x.cg_version, c_select_cols(x.c(), rs))
cpdef Expression sum_cols(Expression x): return Expression.from_cexpr(x.cg_version, c_sum_cols(x.c()))

cpdef Expression sum_batches(Expression x): return Expression.from_cexpr(x.cg_version, c_sum_batches(x.c()))

#expr-opt
cpdef Expression fold_rows(Expression x, unsigned nrows=2): return Expression.from_cexpr(x.cg_version, c_fold_rows(x.c(),nrows))
#expr-expr-opt
# x is scalar or row vector
# y is scalar or row vector
# res = max(0, m - x + y)
cpdef Expression pairwise_rank_loss(Expression x, Expression y, float m=1.0): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_pairwise_rank_loss(x.c(), y.c(), m))
cpdef Expression poisson_loss(Expression x, unsigned y): return Expression.from_cexpr(x.cg_version, c_poisson_loss(x.c(), y))
cpdef Expression huber_distance(Expression x, Expression y, float c=1.345): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_huber_distance(x.c(), y.c(), c))
#expr-unsigned
cpdef Expression kmax_pooling(Expression x, unsigned k): return Expression.from_cexpr(x.cg_version, c_kmax_pooling(x.c(), k))
cpdef Expression pickneglogsoftmax(Expression x, unsigned v): return Expression.from_cexpr(x.cg_version, c_pickneglogsoftmax(x.c(), v))
cpdef Expression pickneglogsoftmax_batch(Expression x, vector[unsigned] vs): return Expression.from_cexpr(x.cg_version, c_pickneglogsoftmax(x.c(), vs))

cpdef Expression kmh_ngram(Expression x, unsigned v): return Expression.from_cexpr(x.cg_version, c_kmh_ngram(x.c(), v))
cpdef Expression pickrange(Expression x, unsigned v, unsigned u): return Expression.from_cexpr(x.cg_version, c_pickrange(x.c(), v, u))
#expr-float
cpdef Expression noise(Expression x, float stddev): return Expression.from_cexpr(x.cg_version, c_noise(x.c(), stddev))
cpdef Expression dropout(Expression x, float p): return Expression.from_cexpr(x.cg_version, c_dropout(x.c(), p))
cpdef Expression block_dropout(Expression x, float p): return Expression.from_cexpr(x.cg_version, c_block_dropout(x.c(), p))
#expr-dim
cpdef Expression reshape(Expression x, tuple d, unsigned int batch_size=1): return Expression.from_cexpr(x.cg_version, c_reshape(x.c(),Dim(d, batch_size)))

cpdef Expression esum(list xs):
    assert xs, 'List is empty, nothing to esum.'
    cdef vector[CExpression] cvec
    cvec = vector[CExpression]()
    cdef Expression x
    for x in xs:
        ensure_freshness(x)
        cvec.push_back(x.c())
    #print(cvec.size(), file=sys.stderr)
    return Expression.from_cexpr(x.cg_version, c_sum(cvec))

cpdef Expression logsumexp(list xs):
    assert xs, 'List is empty, nothing to logsumexp.'
    cdef vector[CExpression] cvec
    cvec = vector[CExpression]()
    cdef Expression x
    for x in xs:
        ensure_freshness(x)
        cvec.push_back(x.c())
    #print(cvec.size(), file=sys.stderr)
    return Expression.from_cexpr(x.cg_version, c_logsumexp(cvec))

cpdef Expression average(list xs):
    assert xs, 'List is empty, nothing to average.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs: 
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_average(cvec))

cpdef Expression emax(list xs):
    assert xs, 'List is empty, nothing to emax.'
    cdef Expression c
    cdef Expression x
    c = xs[0]
    ensure_freshness(c) 
    for x in xs: 
        ensure_freshness(x) 
        c = Expression.from_cexpr(x.cg_version, c_bmax(x.c(),c.c()))
    return c
    #return Expression.from_cexpr(x.cg_version, c_max(cvec))

cpdef Expression concatenate_cols(list xs):
    assert xs, 'List is empty, nothing to concatenate.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs:
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_concat_cols(cvec))

cpdef Expression concatenate(list xs):
    assert xs, 'List is empty, nothing to concatenate.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs:
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_concat(cvec))


cpdef Expression affine_transform(list exprs):
    assert exprs, 'List input to affine_transform must not be empty.'
    cdef Expression e
    cdef vector[CExpression] ves
    for e in exprs:
        ensure_freshness(e) 
        ves.push_back(e.c())
    return Expression.from_cexpr(e.cg_version, c_affine_transform(ves))


# }}}
    
# {{{ RNNS / Builders
# TODO: unify these with inheritance

cdef class _RNNBuilder: # {{{
    cdef CRNNBuilder *thisptr
    cdef RNNState _init_state
    cdef int cg_version 

    def __dealloc__(self):
        del self.thisptr

    cpdef set_dropout(self, float f): self.thisptr.set_dropout(f)
    cpdef disable_dropout(self): self.thisptr.disable_dropout()

    cdef new_graph(self):
        self.thisptr.new_graph(_cg.thisptr[0])
        self.cg_version = _cg.version()

    cdef start_new_sequence(self, es=None):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef vector[CExpression] ces = vector[CExpression]()
        cdef Expression e
        if es:
            for e in es:
                ensure_freshness(e)
                ces.push_back(e.c())
        self.thisptr.start_new_sequence(ces)

    cdef Expression add_input(self, Expression e):
        ensure_freshness(e)
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        return Expression.from_cexpr(self.cg_version, self.thisptr.add_input(e.c()))

    cdef Expression add_input_to_prev(self, CRNNPointer prev, Expression e):
        ensure_freshness(e)
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        return Expression.from_cexpr(self.cg_version, self.thisptr.add_input(prev, e.c()))

    cdef set_h(self, CRNNPointer prev, es=None):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef vector[CExpression] ces = vector[CExpression]()
        cdef Expression e
        if es:
            for e in es:
                ensure_freshness(e)
                ces.push_back(e.c())
        return Expression.from_cexpr(self.cg_version, self.thisptr.set_h(prev, ces))

    cdef set_s(self, CRNNPointer prev, es=None):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef vector[CExpression] ces = vector[CExpression]()
        cdef Expression e
        if es:
            for e in es:
                ensure_freshness(e)
                ces.push_back(e.c())
        return Expression.from_cexpr(self.cg_version, self.thisptr.set_s(prev, ces))

    cdef rewind_one_step(self):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        self.thisptr.rewind_one_step()

    cdef Expression back(self):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        return Expression.from_cexpr(self.cg_version, self.thisptr.back())

    cdef final_h(self):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.final_h()
        for cexp in cexps:
            res.append(Expression.from_cexpr(self.cg_version, cexp))
        return res

    cdef final_s(self):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.final_s()
        for cexp in cexps:
            res.append(Expression.from_cexpr(self.cg_version, cexp))
        return res

    cdef get_h(self, CRNNPointer i):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.get_h(i)
        for cexp in cexps:
            res.append(Expression.from_cexpr(self.cg_version, cexp))
        return res

    cdef get_s(self, CRNNPointer i):
        if self.cg_version != _cg.version(): raise ValueError("Using stale builder. Create .new_graph() after computation graph is renewed.")
        cdef list res = []
        cdef CExpression cexp
        cdef vector[CExpression] cexps = self.thisptr.get_s(i)
        for cexp in cexps:
            res.append(Expression.from_cexpr(self.cg_version, cexp))
        return res

    cpdef RNNState initial_state(self,vecs=None):
        if self.cg_version != _cg.version():
            self.new_graph()
            if vecs is not None:
                self.start_new_sequence(vecs)
            else:
                self.start_new_sequence()
            self._init_state = RNNState(self, -1)
        return self._init_state

    cpdef RNNState initial_state_from_raw_vectors(self,vecs=None):
        if self.cg_version != _cg.version():
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
#}}}

cdef class SimpleRNNBuilder(_RNNBuilder): # {{{
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        if layers > 0:
            self.thisptr = new CSimpleRNNBuilder(layers, input_dim, hidden_dim, model.thisptr[0])
        else:
            self.thisptr = new CSimpleRNNBuilder()
        self.cg_version = -1

    def whoami(self): return "SimpleRNNBuilder"
#}}}
    
cdef class GRUBuilder(_RNNBuilder): # {{{
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        if layers > 0:
            self.thisptr = new CGRUBuilder(layers, input_dim, hidden_dim, model.thisptr[0])
        else:
            self.thisptr = new CGRUBuilder()
        self.cg_version = -1

    def whoami(self): return "GRUBuilder"
# }}}

cdef class LSTMBuilder(_RNNBuilder): # {{{
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        if layers > 0:
            self.thisptr = new CLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr[0])
        else:
            self.thisptr = new CLSTMBuilder()
        self.cg_version = -1

    def whoami(self): return "LSTMBuilder"
# }}}

cdef class VanillaLSTMBuilder(_RNNBuilder): # {{{
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        if layers > 0:
            self.thisptr = new CVanillaLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr[0])
        else:
            self.thisptr = new CVanillaLSTMBuilder()
        self.cg_version = -1

    def whoami(self): return "VanillaLSTMBuilder"
# }}}

cdef class FastLSTMBuilder(_RNNBuilder): # {{{
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, Model model):
        self.thisptr = new CFastLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr[0])
        self.cg_version = -1

    def whoami(self): return "FastLSTMBuilder"
# }}}

class BiRNNBuilder(object):
    """
    Builder for BiRNNs that delegates to regular RNNs and wires them together.  
    
        builder = BiRNNBuilder(1, 128, 100, model, LSTMBuilder)
        [o1,o2,o3] = builder.transduce([i1,i2,i3])
    """
    def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory, builder_layers=None):
        """
        @param num_layers: depth of the BiRNN
        @param input_dim: size of the inputs
        @param hidden_dim: size of the outputs (and intermediate layer representations)
        @param model
        @param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
        @param builder_layers: list of (forward, backward) pairs of RNNBuilder instances to directly initialize layers
        """
        if builder_layers is None:
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
        else:
            self.builder_layers = builder_layers

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

cdef class RNNState: # {{{
    """
    This is the main class for working with RNNs / LSTMs / GRUs.
    Request an RNNState initial_state() from a builder, and then progress from there.
    """
    cdef _RNNBuilder builder
    cdef int state_idx
    cdef RNNState _prev
    cdef Expression _out
    # TODO: should be callable only from C
    def __cinit__(self, _RNNBuilder builder, int state_idx=-1, RNNState prev_state=None, Expression out=None):
        self.builder = builder
        self.state_idx=state_idx
        self._prev = prev_state
        self._out = out

    cpdef RNNState set_h(self, es=None):
        cdef Expression res = self.builder.set_h(CRNNPointer(self.state_idx), es)
        cdef int state_idx = <int>self.builder.thisptr.state()
        return RNNState(self.builder, state_idx, self, res)

    cpdef RNNState set_s(self, es=None):
        cdef Expression res = self.builder.set_s(CRNNPointer(self.state_idx), es)
        cdef int state_idx = <int>self.builder.thisptr.state()
        return RNNState(self.builder, state_idx, self, res)

    cpdef RNNState add_input(self, Expression x):
        cdef Expression res = self.builder.add_input_to_prev(CRNNPointer(self.state_idx), x)
        cdef int state_idx = <int>self.builder.thisptr.state()
        return RNNState(self.builder, state_idx, self, res)

    def add_inputs(self, xs):
        """
        returns the list of states obtained by adding the given inputs
        to the current state, one by one.

        see also transduce(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        states = []
        cur = self
        for x in xs:
            cur = cur.add_input(x)
            states.append(cur)
        return states
    
    cpdef transduce(self, xs):
        """
        returns the list of output Expressions obtained by adding the given inputs
        to the current state, one by one.
        
        see also add_inputs(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        cdef list exprs = []
        cdef Expression res
        cdef Expression x
        cdef int state_idx = self.state_idx
        for x in xs:
            res = self.builder.add_input_to_prev(CRNNPointer(state_idx), x)
            state_idx = <int>self.builder.thisptr.state()
            exprs.append(res)
        return exprs

    #cpdef int state(self): return self.state_idx

    cpdef Expression output(self): return self._out

    cpdef tuple h(self):
        """
        tuple of expressions representing the output of each hidden layer
        of the current step.
        the actual output of the network is at h()[-1].
        """
        return tuple(self.builder.get_h(CRNNPointer(self.state_idx)))

    cpdef tuple s(self):
        """
        tuple of expressions representing the hidden state of the current
        step.

        For SimpleRNN, s() is the same as h()
        For LSTM, s() is a series of of memory vectors, followed the series
                  followed by the series returned by h().
        """
        return tuple(self.builder.get_s(CRNNPointer(self.state_idx)))

    cpdef RNNState prev(self): return self._prev

    def b(self): return self.builder
    #}}}

# StackedRNNState   TODO: do at least minimal testing for this #{{{
cdef class StackedRNNState:
    cdef list states
    cdef StackedRNNState prev
    def __init__(self, list states, StackedRNNState prev=None):
        self.states = states
        self.prev = prev

    cpdef StackedRNNState add_input(self, Expression x):
        cdef list next_states
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
#}}}

# }}}

# {{{ Training 
cdef class SimpleSGDTrainer:
    cdef CSimpleSGDTrainer *thisptr
    def __cinit__(self, Model m, float e0 = 0.1, float edecay = 0.0):
        self.thisptr = new CSimpleSGDTrainer(m.thisptr[0], e0, edecay)
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self, float s=1.0):
        self.thisptr.update(s)
    cpdef update_epoch(self, float r = 1.0):
        self.thisptr.update_epoch(r)
    cpdef status(self):
        self.thisptr.status()
    cpdef set_sparse_updates(self,bool su):
        self.thisptr.sparse_updates_enabled = su
    cpdef set_clip_threshold(self,float thr):
        if thr<=0:
            self.thisptr.clipping_enabled = False
            self.thisptr.clip_threshold = 0.0
        else:
            self.thisptr.clipping_enabled = True
            self.thisptr.clip_threshold = thr
    cpdef get_clip_threshold(self):
        return self.thisptr.clip_threshold

cdef class MomentumSGDTrainer:
    cdef CMomentumSGDTrainer *thisptr
    def __cinit__(self, Model m, float e0 = 0.01, float mom = 0.9, float edecay = 0.0):
        self.thisptr = new CMomentumSGDTrainer(m.thisptr[0], e0, mom, edecay)
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self, float s=1.0):
        self.thisptr.update(s)
    cpdef update_epoch(self, float r = 1.0):
        self.thisptr.update_epoch(r)
    cpdef status(self):
        self.thisptr.status()
    cpdef set_sparse_updates(self,bool su):
        self.thisptr.sparse_updates_enabled = su
    cpdef set_clip_threshold(self,float thr):
        if thr<=0:
            self.thisptr.clipping_enabled = False
            self.thisptr.clip_threshold = 0.0
        else:
            self.thisptr.clipping_enabled = True
            self.thisptr.clip_threshold = thr
    cpdef get_clip_threshold(self):
        return self.thisptr.clip_threshold


cdef class AdagradTrainer:
    cdef CAdagradTrainer *thisptr
    def __cinit__(self, Model m, float e0 = 0.1, float eps = 1e-20, float edecay = 0.0):
        self.thisptr = new CAdagradTrainer(m.thisptr[0], e0, eps, edecay)
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self, float s=1.0):
        self.thisptr.update(s)
    cpdef update_epoch(self, float r = 1.0):
        self.thisptr.update_epoch(r)
    cpdef status(self):
        self.thisptr.status()
    cpdef set_sparse_updates(self,bool su):
        self.thisptr.sparse_updates_enabled = su
    cpdef set_clip_threshold(self,float thr):
        if thr<=0:
            self.thisptr.clipping_enabled = False
            self.thisptr.clip_threshold = 0.0
        else:
            self.thisptr.clipping_enabled = True
            self.thisptr.clip_threshold = thr
    cpdef get_clip_threshold(self):
        return self.thisptr.clip_threshold


cdef class AdadeltaTrainer:
    cdef CAdadeltaTrainer *thisptr
    def __cinit__(self, Model m, float eps = 1e-6, float rho = 0.95, float edecay = 0.0):
        self.thisptr = new CAdadeltaTrainer(m.thisptr[0], eps, rho, edecay)
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self, float s=1.0):
        self.thisptr.update(s)
    cpdef update_epoch(self, float r = 1.0):
        self.thisptr.update_epoch(r)
    cpdef status(self):
        self.thisptr.status()
    cpdef set_sparse_updates(self,bool su):
        self.thisptr.sparse_updates_enabled = su
    cpdef set_clip_threshold(self,float thr):
        if thr<=0:
            self.thisptr.clipping_enabled = False
            self.thisptr.clip_threshold = 0.0
        else:
            self.thisptr.clipping_enabled = True
            self.thisptr.clip_threshold = thr
    cpdef get_clip_threshold(self):
        return self.thisptr.clip_threshold


cdef class AdamTrainer:
    cdef CAdamTrainer *thisptr
    def __cinit__(self, Model m, float alpha = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, eps = 1e-8, float edecay = 0.0 ):
        self.thisptr = new CAdamTrainer(m.thisptr[0], alpha, beta_1, beta_2, eps, edecay)
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self, float s=1.0):
        self.thisptr.update(s)
    cpdef update_epoch(self, float r = 1.0):
        self.thisptr.update_epoch(r)
    cpdef status(self):
        self.thisptr.status()
    cpdef set_sparse_updates(self,bool su):
        self.thisptr.sparse_updates_enabled = su
    cpdef set_clip_threshold(self,float thr):
        if thr<=0:
            self.thisptr.clipping_enabled = False
            self.thisptr.clip_threshold = 0.0
        else:
            self.thisptr.clipping_enabled = True
            self.thisptr.clip_threshold = thr
    cpdef get_clip_threshold(self):
        return self.thisptr.clip_threshold

#}}}

