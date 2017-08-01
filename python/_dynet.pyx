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

from _dynet cimport *
cimport _dynet as dynet


cdef class DynetParams: # {{{
    """This object holds the global parameters of Dynet
    
    This is useful if you want to specify the global dynet parameters (memory, random seed...) programmatically, for example in a notebook.
    
    In order to use this object, you will need to import a different package:
    
    .. code-block:: python

        import _dynet
        # or
        import _gdynet # For GPU
    
    You can then declare and use a :code:`DynetParams` object
    
    .. code-block:: python
        
        # Declare a DynetParams object
        dyparams = dy.DynetParams()
        # Fetch the command line arguments (optional)
        dyparams.from_args()
        # Set some parameters manualy (see the command line arguments documentation)
        dyparams.set_mem(2048)
        dyparams.set_random_seed(666)
        # Initialize with the given parameters
        dyparams.init() # or init_from_params(dyparams)
        
    Don't forget to initialize with :code:`dyparams.init()`, otherwise dynet will raise an error.
    """
    cdef CDynetParams cparams

    def __init__(self):
        pass

    cpdef from_args(self, shared_parameters=None):
        """Gets parameters from the command line arguments
        
        You can still modify the parameters after calling this.
        See the documentation about command line arguments for more details
        
        Keyword Args:
            shared_parameters([type]): [description] (default: None)
        """
        cdef int argc = len(sys.argv)
        cdef char** c_argv
        args = [bytearray(x, encoding="utf-8") for x in sys.argv]
        c_argv = <char**>malloc(sizeof(char*) * len(args)) # TODO check failure?
        for idx, s in enumerate(args):
            c_argv[idx] = s

        if shared_parameters is None:
            self.cparams = dynet.extract_dynet_params(argc,c_argv, 0)
        else:
            if shared_parameters == 0: shared_parameters = 1
            self.cparams = dynet.extract_dynet_params(argc,c_argv, shared_parameters)
        free(c_argv)

    cpdef init(self):
        """Initialize dynet with the current dynetparams object.
        
        This is one way, you can't uninitialize dynet
        """
        dynet.initialize(self.cparams)

    cpdef set_mem(self, unsigned mem):
        """Set the memory allocated to dynet
        
        The unit is MB
        
        Args:
            mem(number): memory size in MB
        """
        self.cparams.mem_descriptor = str(mem).encode()

    cpdef set_random_seed(self, unsigned random_seed):
        """Set random seed for dynet
        
        Args:
            random_seed(number): Random seed
        """
        self.cparams.random_seed = random_seed

    cpdef set_autobatch(self, bool autobatch):
        """Activate autobatching
        
        Args:
            autobatch(bool): Set to :code:`True` to activate autobatching
        """
        if autobatch:
            self.cparams.autobatch = 1
        else:
            self.cparams.autobatch = 0

    cpdef set_autobatch_debug(self, bool autobatch_debug):
        """Activate autobatching debug
        
        Args:
            autobatch(bool): Set to :code:`True` to activate autobatching debug
        """
        if autobatch_debug:
            self.cparams.autobatch_debug = 1
        else:
            self.cparams.autobatch_debug = 0

    cpdef set_weight_decay(self, float weight_decay):
        """Set weight decay parameter
        
        Args:
            weight_decay(float): weight decay parameter
        """
        self.cparams.weight_decay = weight_decay

    cpdef set_shared_parameters(self, bool shared_parameters):
        """Shared parameters
        
        Args:
            shared_parameters(bool): shared parameters
        """
        self.cparams.shared_parameters = shared_parameters

    cpdef set_requested_gpus(self, int requested_gpus):
        """Number of requested gpus
        
        Currently only 1 is supported
        
        Args:
            requested_gpus(number): number of requested gpus
        """
        self.cparams.requested_gpus = requested_gpus
        self.cparams.ngpus_requested = True
        self.cparams.ids_requested = False
    
    cpdef set_gpu_mask(self, list gpu_mask):
        cdef vector[int] cgpu_mask
        for i in gpu_mask:
            if(i!=0 and i!=1):
                raise ValueError('gpu_mask should only contain 0 and 1s')
            cgpu_mask.push_back(i)
        self.cparams.gpu_mask = cgpu_mask
        self.cparams.ngpus_requested = False
        self.cparams.ids_requested = True
# DynetParams }}}

# Initialization {{{
def init(shared_parameters=None):
    """Initialize dynet
    
    Initializes dynet from command line arguments. Do not use after 
        
        import dynet

    only after 

        import _dynet / import _gdynet
    
    Keyword Args:
        shared_parameters(bool): [description] (default: None)
    """
    params=DynetParams()
    params.from_args(shared_parameters)
    params.init()

def init_from_params(DynetParams params):
    """Initialize from DynetParams
    
    Same as 

        params.init()
    
    Args:
        params(DynetParams): dynet parameters
    """
    params.init()
# }}}

# Dimensions {{{
cdef CDim Dim(dim, unsigned int batch_size=1):
    """Get dynet Dim from tuple
    

    Args:
        dim(tuple): Dimensions as a tuple
        batch_size(number): Batch size (default: 1)
    
    Returns:
        CDim: Dynet dimension
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

cdef tuple c_dim_as_dim(CDim d):
    """
    Returns a tuple (dims,batch_dim) where dims is the tuple of dimensions of each batch element
    """
    dim = tuple([d[i] for i in range(d.ndims())])
    dim= (dim,d.batch_elems())
    return tuple(dim)

cdef tuple c_dim_as_shape(CDim d,bool force_batch=False):
    dim = [d[i] for i in range(d.ndims())]
    if force_batch or d.batch_elems()>1: dim.append(d.batch_elems())
    return tuple(dim)

cdef CDim shape_as_c_dim(tuple d,bool batched = False):
    if batched:
        dim = d[:-1] if len(d) > 1 else (1,)
        batch_size= d[-1]
    else:
        dim = d
        batch_size = 1
    return Dim(dim,batch_size=batch_size)
# }}}

# IO {{{

cdef _save_one(datafname, fh, obj):
    if isinstance(obj, Parameters):
        pickle.dump(("Parameters", obj.name()), fh)
        obj.save(datafname,append=True)
    elif isinstance(obj, LookupParameters):
        pickle.dump(("LookupParameters", obj.name()), fh)
        obj.save(datafname,append=True)
    else:
        pickle.dump((obj.__class__, obj.spec, obj.param_collection().name()), fh)
        obj.param_collection().save(datafname,append=True)

cdef _load_one(datafname, fh, model):
    o = pickle.load(fh)
    if o[0] == 'Parameters':
        p = model.load_param(datafname, o[1])
        return p
    if o[0] == 'LookupParameters':
        p = model.load_lookup_param(datafname, o[1])
        return p
    else:
        cls, spec, name = o
        obj = cls.from_spec(spec, model)
        obj.param_collection().populate(datafname, name)
        return obj

cpdef save(basename, lst):
    """Saves a list of parameters, lookup parameters and builder objects to disk.

    Args:
        basename (string): The base-name of the files to save. Two files will be created: `basename.data` and `basename.meta`.
        lst      (list):  A list of objects to save (see below).


    Example:

        import dynet as dy

        pc = dy.ParameterCollection()
        W = pc.add_parameters((100,50))
        E = pc.add_lookup_parameters((1000,50))
        builder = dy.LSTMBuilder(2, 50, 50, pc)

        dy.save("model", [E, builder, W])

        # then, when loading:
        pc = dy.ParameterCollection()
        E2, builder2, W2 = dy.load("model", pc)

    
    What can be saved:
        Each object in `lst` must be one of the following:
        
        (1) Parameter
        (2) LookupParameter
        (3) one of the built-in types (CompactVanillaLSTMBuilder, VanillaLSTMBuilder, LSTMBuilder, GRUBuilder,
                                       SimpleRNNBuilder, BiRNNBuilder)
        (4) a type adhering to the following interface:
            
            - has a `param_collection()` method returning a ParameterCollection object with the
              parameters in the object.
            - has a `.spec` property with picklable items describing the object
            - has a `.from_spec(spec, model)` static method that will create and return a
              new instane with the needed parameters/etc in the model.

        Note, the built-in types in (3) above can be saved/loaded this way simply because 
        they support this interface.

        behind the scenes:
        
        - for each item, we write to `.meta`:
            if its a Parameters/ParameterCollection: 
                its type and full name.
            if its a builder:
                its class, its spec, the full name of its parameters collection.
        - the associated parameters/sub-collection is then saved to `.data`
    """
    open(basename+".data","w").close() # delete current
    fh = open(basename+".meta","wb")
    for item in lst:
        _save_one(basename+".data", fh, item)
    fh.close()

cpdef load(basename, params):
    """Loads a list of parameters, lookup parameters and builder objects from disk.
    The loaded objects are added to the supplied params collection, and returned.

    Args:
        basename (string):  The basename to read from. 
                            This is the same string that was used when saving the objects.
        params   (dynet.ParameterCollection): A ParameterCollection to add the loaded objects to.

    Returns:
        A list of parameters, lookup parameters and builder objects, in the same order they
        were passed to the save function.


    Example:
        import dynet as dy

        pc = dy.ParameterCollection()
        W = pc.add_parameters((100,50))
        E = pc.add_lookup_parameters((1000,50))
        builder = dy.LSTMBuilder(2, 50, 50, pc)

        dy.save("model", [E, builder, W])

        # then, when loading:
        pc = dy.ParameterCollection()
        E2, builder2, W2 = dy.load("model", pc)
    """
    fh = open(basename+".meta","rb")
    res = []
    while True:
        try:
            obj = _load_one(basename+".data", fh, params)
        except EOFError: break
        res.append(obj)
    fh.close()
    return res
# }}}

cdef c_tensor_as_np(CTensor &t):
    # TODO: make more efficient, with less copy
    arr = np.array(c_as_vector(t))
    dim = c_dim_as_shape(t.d)
    return arr.reshape(dim,order='F')

cdef c_index_tensor_as_np(CIndexTensor &t):
    # TODO: make more efficient, with less copy
    arr = np.array(c_index_tensor_as_vector(t))
    dim = c_dim_as_shape(t.d)
    return arr.reshape(dim,order='F')


# Initializers {{{
cdef class PyInitializer:
    """
    Base class for parameter initializer
    """
    cdef CParameterInit *initializer
    def __init__(self):
        assert(False),"Do not create PyInitializer directly."
    def __dealloc__(self):
        del self.initializer

cdef class NormalInitializer(PyInitializer):
    """Initialize the parameters with a gaussian distribution
    
    Keyword Arguments:
        mean (number): Mean of the distribution (default: 0)
        var (number): Variance of the distribution (default: 1)
    """
    def __init__(self, float mean=0, var=1):
        self.initializer = new CParameterInitNormal(mean, var)

cdef class UniformInitializer(PyInitializer):
    """Initialize the parameters with a uniform distribution
    
    Args:
        scale (number): Parmeters are sampled from :math:`\mathcal U([-\\texttt{scale},\\texttt{scale}])`
    """
    def __init__(self, float scale):
        self.initializer = new CParameterInitUniform(scale)

cdef class ConstInitializer(PyInitializer):
    """Initialize the parameters with a constant value
    
    Args:
        c (number): Value to initialize the parameters
    """
    def __init__(self, float c):
        self.initializer = new CParameterInitConst(c)

cdef class IdentityInitializer(PyInitializer):
    """Initialize the parameters as the identity
    
    Only works with square matrices
    """
    def __init__(self):
        self.initializer = new CParameterInitIdentity()

cdef class GlorotInitializer(PyInitializer):
    """Initializes the weights according to `Glorot & Bengio (2011) <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ 
    
    If the dimensions of the parameter matrix are :math:`m,n`, the weights are sampled from :math:`\mathcal U([-g\sqrt{\\frac{6}{m+n}},g\sqrt{\\frac{6}{m+n}}])`
    
    The gain :math:`g` depends on the activation function : 

    * :math:`\\text{tanh}` : 1.0
    * :math:`\\text{ReLU}` : 0.5
    * :math:`\\text{sigmoid}` : 4.0
    * Any smooth function :math:`f` : :math:`\\frac{1}{f'(0)}`
    
    Keyword Arguments:
        is_lookup (bool): Whether the parameter is alookup parameter (default: False)
        gain (number): Gain (Depends on the activation function) (default: 1.0)
    """
    def __init__(self, bool is_lookup=False,float gain=1.0):
        self.initializer = new CParameterInitGlorot(is_lookup,gain)

cdef class SaxeInitializer(PyInitializer):
    """Initializes according to `Saxe et al. (2014) <https://arxiv.org/abs/1312.6120>`_

    Initializes as a random orthonormal matrix (unimplemented for GPU)
        Keyword Arguments:
            scale (number): scale to apply to the orthonormal matrix
    """
    def __init__(self,scale=1.0):
        self.initializer = new CParameterInitSaxe(scale)

cdef class FromFileInitializer(PyInitializer):
    """Initialize parameter from file
    
    Args:
        fname (str): File name
    """
    def __init__(self, string fname):
        self.initializer = new CParameterInitFromFile(fname)

cdef class NumpyInitializer(PyInitializer):
    """Initialize from numpy array

    Alternatively, use :code:`ParameterCollection.parameters_from_numpy()`
    
    Args:
        array (np.ndarray): Numpy array
    """
    def __init__(self, array):
        self.initializer = new CParameterInitFromVector(self.vec_from_array(array))

    cdef vector[float] vec_from_array(self, arr): # TODO make efficient
        cdef vector[float] vals
        shape = arr.shape
        arr = arr.flatten(order='F')
        for i in xrange(arr.size):
            vals.push_back(arr[i])
        return vals
# }}}

# {{{ ParameterCollection / Parameters 
cdef class Parameters: # {{{
    """Parameters class
    
    Parameters are things that are optimized. in contrast to a system like Torch where computational modules may have their own parameters, in DyNet parameters are just parameters.
    """
    cdef CParameters thisptr # TODO: no longer pointer
    cdef int _version
    cdef Expression _expr
    cdef int _const_version
    cdef Expression _const_expr
    def __cinit__(self):
        self._version = -1
    @staticmethod
    cdef wrap_ptr(CParameters ptr):
        self = Parameters()
        self.thisptr = ptr
        return self

    # TODO docs
    def save(self, fname, key="",append=False):
        self.write_to_textfile(fname, key,append)

    # TODO docs
    def populate(self, fname, key):
        """Populate the values of this Parameters object from
        the parameter named `key` in the file `fname`.
        The sizes of saved parameters and this object must match.

        Args:
            fname (string): the name of a file to load from.
            key   (string): the parameter to read from the file.
        """
        self.populate_from_textfile(fname, key)

    # TODO docs
    def write_to_textfile(self, fname, key="", bool append=False):
        cdef CTextFileSaver *saver
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        saver = new CTextFileSaver(_fname, append=append)
        saver.save(self.thisptr,_key)
        del saver

    # TODO docs
    def populate_from_textfile(self, string fname, string key=""):
        cdef CTextFileLoader *loader
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        loader = new CTextFileLoader(_fname)
        loader.populate(self.thisptr, _key)
        del loader

    cpdef shape(self):
        """Returns shape of the parameter
        
        Returns:
            tuple: Shape of the parameter
        """
        return c_dim_as_shape(self.thisptr.get_storage().dim)

    cpdef as_array(self):
        """Return as a numpy array.
        
        Returns:
            np.ndarray: values of the parameter
        """
        cdef CTensor t
        return c_tensor_as_np(self.thisptr.get_storage().values)

    cpdef grad_as_array(self):
        """Return gradient as a numpy array.
        
        Returns:
            np.ndarray: values of the gradient w.r.t. this parameter
        """
        cdef CTensor t
        return c_tensor_as_np(self.thisptr.get_storage().g)
    
    cpdef clip_inplace(self, float left, float right):
        """Clip the values in the parameter to a fixed range [left, right] (in place)
        
        Args:
            arr(np.ndarray): Scale
        """
        self.thisptr.clip_inplace(left, right)
        
    # TODO: make more efficient
    cpdef set_value(self, arr):
        """Set value of the parameter

        """
        shape = arr.shape
        if self.shape() != shape:
            raise ValueError("Shape of values and parameter don't match in Parameters.set_value")
        arr = arr.flatten(order='F')
        self.thisptr.set_value(arr)

    cpdef zero(self):
        """Set the parameter to zero

        """
        self.thisptr.zero()

    cpdef scale(self,float s):
        """Scales the parameter

        Args:
            s(float): Scale

        """
        self.thisptr.scale(s)

    cpdef scale_gradient(self,float s):
        """Scales the gradient

        Args:
            s(float): Scale

        """
        self.thisptr.scale_gradient(s)


    cpdef bool is_updated(self):
        """check whether the parameter is updated or not
        
        Returns:
            bool: Update status
        """
        return self.thisptr.is_updated()

    cpdef set_updated(self, bool b):
        """Set parameter as "updated"
        
        Args:
            b(bool): updated status
        """
        self.thisptr.set_updated(b)

    cpdef name(self):
        """
        Return the full name of this parameter.
        """
        return self.thisptr.get_fullname().decode("utf8")

    cpdef Expression expr(self, bool update=True):
        """Returns the parameter as an expression

        This is the same as calling

            dy.parameter(param)
        
        Args:
            update(bool): If this is set to False, the parameter won't be updated during the backward pass
        Returns:
            Expression: Expression of the parameter
        """
        if update:
            if cg_version() != self._version:
                self._version = cg_version()
                self._expr = Expression.from_cexpr(_cg.version(), c_parameter(_cg.thisptr[0], self.thisptr))
            return self._expr
        else:
            if cg_version() != self._const_version:
                self._const_version = cg_version()
                self._const_expr = Expression.from_cexpr(_cg.version(), c_const_parameter(_cg.thisptr[0], self.thisptr))
            return self._const_expr

# Parameters }}}

cdef class LookupParameters: # {{{
    """LookupParameters represents a table of parameters.

    They are used to embed a set of discrete objects (e.g. word embeddings). These are sparsely updated.

    """
    cdef CLookupParameters thisptr # TODO: no longer pointer
    cdef int _version
    cdef Expression _expr
    def __cinit__(self):
        self._version = -1
    @staticmethod
    cdef wrap_ptr(CLookupParameters ptr):
        self = LookupParameters()
        self.thisptr = ptr
        return self

    # TODO docs
    def save(self, fname, key="", append=False):
        """Save the values of this LookupParameters object to a particular file.

        TODO: more docs. Refer to the tutorial for more info for now

        Args:
            fname (string): the name of a file to save to.
            key   (string): TODO
        """
        self.write_to_textfile(fname, key, append)
    def populate(self, fname, key=""):
        """Populate the values of this LookupParameters object from
        the parameter named `key` in the file `fname`.
        The sizes of saved parameters and this object must match.

        Args:
            fname (string): the name of a file to load from.
            key   (string): the parameter to read from the file.
        """
        self.populate_from_textfile(fname, key)

    def write_to_textfile(self, fname, key="", bool append=False):
        cdef CTextFileSaver *saver
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        saver = new CTextFileSaver(_fname, append=append)
        saver.save(self.thisptr,_key)
        del saver

    def populate_from_textfile(self, fname, key=""):
        cdef CTextFileLoader *loader
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        loader = new CTextFileLoader(_fname)
        loader.populate(self.thisptr, _key)
        del loader

    cpdef init_from_array(self, arr):
        """Initializes the values according to a numpy array

        Preferably uses ParameterCollection.lookup_parameter_from_numpy when possible

        Args:
            arr (np.array): numpy array of shape :code:`(num_lookups,...)`
        """
        if len(arr) > self.thisptr.get_storage().values.size():
            raise Exception("too many rows")
        if arr.shape[1] != self.thisptr.get_storage().values[0].d.rows():
            raise Exception("dim mismatch")
        cdef vector[float] r
        for i,row in enumerate(arr):
            self.init_row(i, row)

    cpdef shape(self):
        """Returns shape of the lookup parameter

        The first dimension is the lookup dimension

        Returns:
            tuple: Shape of the parameter
        """
        return c_dim_as_shape(self.thisptr.get_storage().all_dim)

    def __getitem__(self, int i):
        """
        Same as :code:`dynet.lookup`
        """
        return lookup(self, i)

    cpdef batch(self, vector[unsigned] i):
        """Returns a batched expression based on looked up indices

        This does the same as :code:`dynet.lookup_batch`
        
        Args:
            i (list): list of indices

        Returns:
            dynet.Expression: Batched expression fo batch dimension :code:`len(i)`
        """
        return lookup_batch(self, i)

    cpdef init_row(self, unsigned i, vector[float] row):
        """Initialize one row with values
        
        Args:
            i    (int): index
            row (list): values
        """
        self.thisptr.initialize(i, row)

    cpdef as_array(self):
        """Return as a numpy array.

        The first dimension is the lookup dimension

        Returns:
            np.array: Values
        """
        cdef vector[CTensor] vals
        vals = self.thisptr.get_storage().values
        return np.vstack([c_tensor_as_np(t).reshape(1,-1,order='F') for t in vals])

    cpdef grad_as_array(self):
        """Return gradients as a numpy array.

        The first dimension is the lookup dimension

        Returns:
            np.array: gradient values
        """
        cdef vector[CTensor] grads
        grads = self.thisptr.get_storage().grads
        return np.vstack([c_tensor_as_np(t).reshape(1,-1,order='F') for t in grads])
    
    cpdef scale(self,float s):
        """Scales the parameter

        Args:
            s(float): Scale

        """
        self.thisptr.scale(s)

    cpdef scale_gradient(self,float s):
        """Scales the gradient

        Args:
            s(float): Scale

        """
        self.thisptr.scale_gradient(s)
        
    cpdef Expression expr(self,bool update=True):
        """Returns an expression for the whole parameter

        Same as :code:`dynet.parameter`

        Args:
            update(bool): If this is set to False, the parameter won't be updated during the backward pass
        Returns:
            Expression: Expression of the parameter
        """
        if cg_version() != self._version:
            self._version = cg_version()
            if update:
                self._expr = Expression.from_cexpr(_cg.version(), c_parameter(_cg.thisptr[0], self.thisptr))
            else:
                self._expr = Expression.from_cexpr(_cg.version(), c_const_parameter(_cg.thisptr[0], self.thisptr))
        return self._expr

    cpdef zero(self):
        """Set all values to zero
        """
        self.thisptr.zero()

    cpdef bool is_updated(self): return self.thisptr.is_updated()
    cpdef set_updated(self, bool b): self.thisptr.set_updated(b)

    cpdef name(self):
        """
        Return the full name of this lookup parameter.
        """
        return self.thisptr.get_fullname().decode("utf8")
# }}}

cdef class ParameterCollection: # {{{
    """
    A ParameterCollection holds Parameters. Use it to create, load and save parameters.

    (It used to be called Model in previous versions of DyNet, and Model is still an alias for ParameterCollection.)

    A ParameterCollection is a container for Parameters and LookupParameters.

    dynet.Trainer objects take ParameterCollection objects that define which parameters
    are being trained.
    
    The values of the parameters in a collection can be persisted to and loaded from files.

    Hierarchy:
        The parameter collections can be nested, where each collection can hold zero or more
        sub-collection, which are also ParameterCollection objects. Each (sub-)collection contains
        the parameters in it and in all the (sub-)collections below it.

    Naming:
        Parameters, LookupParameters and ParameterCollections have associated string names.
        The names can be accessed using the `.name()` method. 
        
        The names are used for identifying the parameters and the collection hierarchy 
        when loading from disk, and in particular when loading only a subset of the objects 
        in a saved file.

        The name of a parameter, lookup parameter or sub-collection is unique within
        a ParameterCollection, and reflects the hierarchy structure.

        One can supply an optional informative name when creating the parameter or
        sub-collection.  The supplied names are then appended with running index to
        avoid name clashes. The `.name()` method returns the full name of an object,
        including the appended index and its location within the collection hierarchy.
        The user-supplied names cannot inclue the characters `/` (which is used as a hierarchy
        separator) or :code:`_` (which is used as an index separator).
    """
    cdef CModel thisptr  # Not a pointer...
    def __cinit__(self, ):
        pass

    def __init__(self):
        pass

    @staticmethod
    cdef wrap(CModel m):
        self = ParameterCollection()
        self.thisptr = m
        return self

    def save(self, fname, name="",append=False):
        """Save the values of all parameters in this collection to file.

        Args:
            fname (string): file name to save into.
        """
        self.write_to_textfile(fname, name ,append)

    def populate(self, fname, key=""):
        """Populate the values of all parameters in this collection from file.

        This only populates the values of existing parameters, and does not add parameters to the collection.
        Thus, the content of the file and the parameters in this collection must match.
        One should make sure to add to the collection the same parameters (and in the same order) before calling
        populate, as the ones that were added before calling save.

        Args:
            fname (string): file name to read parameter values from.
        """
        self.populate_from_textfile(fname, key)

    cpdef load_param(self, fname, key):
        """Loads a named parameter from a file, adds it to the collection,
        and returns the loaded parameter.

        Args:
            fname (string): the file name to read from.
            key   (string): the full-name of the parameter to read.

        Returns:
            (dynet.Parameters) The Parameters object.
        """
        cdef CTextFileLoader *loader
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        loader = new CTextFileLoader(_fname)
        p = Parameters.wrap_ptr(loader.load_param(self.thisptr, _key))
        del loader
        return p

    cpdef load_lookup_param(self, fname, key):
        """Loads a named lookup-parameter from a file, adds it to the collection,
        and returns the loaded parameter.

        Args:
            fname (string): the file name to read from.
            key   (string): the full-name of the lookup parameter to read.

        Returns:
            (dynet.LookupParameters) The LookupParameters object.
        """
        cdef CTextFileLoader *loader
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        loader = new CTextFileLoader(_fname)
        p = LookupParameters.wrap_ptr(loader.load_lookup_param(self.thisptr, _key))
        del loader
        return p

    # TODO docs
    def write_to_textfile(self, fname, key="",append=False):
        cdef CTextFileSaver *saver
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        saver = new CTextFileSaver(_fname, append=append)
        saver.save(self.thisptr,_key)
        del saver

    # TODO docs
    def populate_from_textfile(self, fname, key=""):
        cdef CTextFileLoader *loader
        cdef string _fname = <string> fname.encode("utf8")
        cdef string _key = <string> key.encode("utf8")
        loader = new CTextFileLoader(_fname)
        loader.populate(self.thisptr, _key)
        del loader

    # TODO: for debug, remove
    cpdef pl(self): return self.thisptr.parameters_list().size()

    cpdef parameters_from_numpy(self, array,string name=""):
        """Create parameter from numpy array
        
        Args:
            array (np.ndarray): Numpy array
            name  (string): optional name for this parameter.
        
        Returns:
            (dynet.Parameters): Parameter
        """
        dim = array.shape
        cdef CParameters p = self.thisptr.add_parameters(Dim(dim), deref(NumpyInitializer(array).initializer),name)
        cdef Parameters pp = Parameters.wrap_ptr(p)
        return pp

    # TODO this may fail with >2 dim arrays.
    cpdef lookup_parameters_from_numpy(self, array, string name=""):
        """Create LookupParameters from numpy array
        
        Args:
            array (np.ndarray): Numpy array. rows: vocab_size, cols: dims.
            name  (string): optional name for this parameter.
        
        Returns:
            (dynet.LookupParameters): LookupParameter
        """
        vocab_size = array.shape[0]
        emb_dim = array.shape[1:]
        init = NumpyInitializer(array.T)
        cdef CLookupParameters p = self.thisptr.add_lookup_parameters(vocab_size, Dim(emb_dim), deref(init.initializer), name)
        cdef LookupParameters pp = LookupParameters.wrap_ptr(p)
        return pp

    cpdef add_parameters(self, dim, PyInitializer init=None, string name=""):
        """Add a parameter to the ParameterCollection
        
        Args:
            dim (tuple): Shape of the parameter
        
        Keyword Arguments:
            init (dynet.PyInitializer): Initializer (default: GlorotInitializer)
            name (string)             : Optional name for this parameter (default: "")
        
        Returns:
            (dynet.Parameters): Created Parameter
        """
        assert(isinstance(dim,(tuple,int)))
        cdef CParameters p
        cdef CParameterInit *initializer
        if init is None:
            init = GlorotInitializer()
        initializer = init.initializer
        p = self.thisptr.add_parameters(Dim(dim), deref(initializer), name)
        cdef Parameters pp = Parameters.wrap_ptr(p)
        return pp

    cpdef add_lookup_parameters(self, dim, PyInitializer init=None, string name=""):
        """Add a lookup parameter to the ParameterCollection
        
        Args:
            dim (tuple): Shape of the parameter. The first dimension is the vocab size
        
        Keyword Arguments:
            init (dynet.PyInitializer): Initializer (default: GlorotInitializer)
            name (string)             : Optional name for this parameter (default: "")
        
        Returns:
            (dynet.LookupParameters): Created LookupParameter
        """
        assert(isinstance(dim, tuple))
        cdef int nids = dim[0]
        rest = tuple(dim[1:])
        if init is None:
            init = GlorotInitializer(True)
        initializer = init.initializer
        cdef CLookupParameters p = self.thisptr.add_lookup_parameters(nids, Dim(rest), deref(initializer), name)
        cdef LookupParameters pp = LookupParameters.wrap_ptr(p)
        return pp

    cpdef add_subcollection(self, name=None):
        """Creates a sub-collection of the current collection, and returns it.
        
        A sub-collection is simply a ParameterCollection object which is tied to a
        parent collection. ParameterCollections can be nested to arbitraty depth.

        Sub-collections are used for grouping of parameters,
        for example if one wants to train only a subset of the parameters, one
        can add them in a subcollection and pass the subcollection to a trainer.
        Similarly, for saving (or loading) only some of the parameters, one can save/populate
        a sub-collection.

        Sub-collections are used inside builder objects (such as the LSTMBuilder):
        The builder creates a local sub-collection and adds parameters to it instead
        of to the global collection that is passed to it in the constructor.
        This way, the parameters participating in the builder are logically grouped,
        and can be saved/loaded/trained seperately if needed.

        Args:
            name (string): an optional name for the sub-collection.

        Keyword Arguments:
            name (string)             : Optional name for this sub-collection (default: "")

        Returns:
            (dynet.ParameterCollection) a parameter collection.
        """
        if name is None: return ParameterCollection.wrap(self.thisptr.add_subcollection("".encode()))
        else: return ParameterCollection.wrap(self.thisptr.add_subcollection(name.encode()))

    cpdef name(self):
        """
        Return the full name of this collection.
        """
        return self.thisptr.get_fullname().decode("utf8")

# Alias Model and ParameterCollection
Model=ParameterCollection

# }}}

# }}}


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

# {{{ Computation Graph 
cdef int SECRET = 923148
cdef ComputationGraph _cg = ComputationGraph(SECRET)

def cg_version(): 
    """
    Varsion of the current computation graph
    """
    return _cg._cg_version

def renew_cg(immediate_compute=False, check_validity=False, autobatching=None): 
    """
    Renew the computation graph.

    Call this before building any new computation graph
    """
    return _cg.renew(immediate_compute, check_validity, autobatching)

def print_text_graphviz(): return _cg.print_graphviz()
def cg_checkpoint(): 
    """
    Saves the state of the computation graph
    """
    _cg.checkpoint()
def cg_revert():
    """
    Revert the computation graph state to the previous checkpoint
    """
    _cg.revert()

cpdef ComputationGraph cg():
    """
    Get the current ComputationGraph
    """
    global _cg
    return _cg

cdef class ComputationGraph:
    """
    Computation graph object

    While the ComputationGraph is central to the inner workings of DyNet, from the user's perspective, the only responsibility is to create a new computation graph for each training example.
    """
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

    cpdef renew(self, immediate_compute=False, check_validity=False, autobatching=None):
        """
        Same as :code:`dynet.renew_cg()`
        """
        del self.thisptr
        if autobatching is None:
            self.thisptr = new CComputationGraph()
        else:
            self.thisptr = new CComputationGraph(autobatching)
        if immediate_compute: self.thisptr.set_immediate_compute(immediate_compute)
        if check_validity: self.thisptr.set_check_validity(check_validity)
        self._inputs = []
        self._cg_version += 1
        return self

    cpdef version(self): 
        """
        Same as :code:`dynet.cg_version()`
        """
        return self._cg_version

    def parameters(self, Parameters params):
        """
        Same as :code:`dynet.parameters(params)`
        """
        cdef Expression result
        result = Expression.from_cexpr(self._cg_version, c_parameter(self.thisptr[0], params.thisptr))
        return result

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

    cpdef backward(self, VariableIndex index, bool full=False):
        self.thisptr.backward(index, full)

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
    def inputMatrixLiteral(self, vector[float] v, tuple d, int batch_size=1):
        return _vecInputExpression(self, v, d,batch_size)
    def inputSparseTensor(self, vector[unsigned] idxs, vector[float] v, tuple dim, int batch_size=1, float defval=0):
        return _sparseInputExpression(self, idxs, v, dim, batch_size, defval)
    cdef lookup(self, LookupParameters p, unsigned v = 0, update=True):
        return _lookupExpression(self, p, v, update)
    cdef lookup_batch(self, LookupParameters p, vector[unsigned] vs, update=True):
        return _lookupBatchExpression(self, p, vs, update)
    cdef outputPicker(self, Expression e, unsigned v=0, unsigned dim=0):
        r = _pickerExpression(self, e, v, dim)
        return r
    cdef outputBatchPicker(self, Expression e, vector[unsigned] vs, unsigned dim=0):
        r = _pickerBatchExpression(self, e, vs, dim)
        return r
# }}}

cdef class Tensor: #{{{
    """Tensor class

    A Tensor is a value object that is kept on the computation device (GPU or CPU).
    It can be converted to a python object (a numpy array), which involves a device-to-python
    copy. More importantly, some operations (ie argmax) can be applied on the tensor on-device,
    reducing the memory copy and potentially benefitting from more efficiency.
    """
    cdef CTensor t
    cdef CIndexTensor lt
    cdef int type
    def __cinit__(self):
        self.type = -1 # 0 if Tensor, 1 is IndexTensor
        pass

    @staticmethod
    cdef wrap_ctensor(CTensor t):
        self = Tensor()
        self.t = t
        self.type = 0
        return self

    @staticmethod
    cdef wrap_cindextensor(CIndexTensor t):
        self = Tensor()
        self.lt = t
        self.type = 1
        return self

    def __str__(self):
        return "<Dynet Tensor:%s>" % ("int" if self.type==1 else "float")

    cpdef as_numpy(self):
        """Converts the tensor values into a numpy array.

        Returns:
            np.ndarray: numpy array of values
        """
        if self.type == 0:
            return np.array(c_tensor_as_np(self.t))
        elif self.type == 1:
            return np.array(c_index_tensor_as_np(self.lt))
        raise ValueError("Improperly Intialized Tensor")

    cpdef argmax(self, unsigned dim=0, unsigned num=1):
        """Calculate the index of the maximum value.

        Assumes that rach row represents a probability distribution.

        Keyword Args:
            dim(integer): which dimension to take the argmax over
            num(integer): the number of kmax values
        
        Returns:
            A newly allocated Tensor consisting of argmax IDs. The length of the
        """
        return Tensor.wrap_cindextensor(CTensorTools.argmax(self.t, dim, num))

    cpdef categorical_sample_log_prob(self, unsigned dim=0, unsigned num=1):
        """Calculate samples from a log probability

        Assumes each row in the tensor represents a log probabiity distribution
        
        Keyword Args:
            dim(integer): Which dimension to take the sample over
            num(integer): num The number of samples for each row

        Returns:
            A newly allocated Tensor consisting of argmax IDs. The length of the
            dimension "dim" will be "num", consisting of the appropriate IDs.
        """
        return Tensor.wrap_cindextensor(CTensorTools.categorical_sample_log_prob(self.t, dim, num))
# Tensor }}}

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
    """Expressions are the building block of a Dynet computation graph.
    
    Expressions are the main data types being manipulated in a DyNet program. Each expression represents a sub-computation in a computation graph.
    """
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

    def dim(self):
        """Dimension of the expression

        Returns a tuple (dims,batch_dim) where dims is the tuple of dimensions of each batch element
        
        Returns:
            tuple: dimension
        """
        cdef CDim d;
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        d=self.c().dim()
        return c_dim_as_dim(d)
        # return (d.size(), d.rows(), d.cols(), d.batch_elems())

    def __repr__(self):
        return str(self)
    def __str__(self):
        """Returns a string representation of the expression
        
        The format is "expression [expression id]/[computation graph id]"
        """
        return "expression %s/%s" % (<int>self.vindex, self.cg_version)

    # __getitem__ and __getslice__ in one for python 3 compatibility
    def __getitem__(self, index):
        """Access elements of the expression by rows
        
        This supports slices as well.
        Example usage :

            x = dy.inputTensor(np.arange(9).reshape(3,3))
            # x = [[1, 2, 3],
            #      [4, 5, 6],
            #      [7, 8, 9]] 
            y = x[1]
            # y = [4, 5, 6]
            z = x[0:1] 
            # z = [[1, 2, 3],
            #      [4, 5, 6]] 
        
        Args:
            index(int,slice): Slice or index
        
        Returns:
            Expression: Slice of the expression
        
        Raises:
            IndexError: If the indices are too large
            ValueError: In case of improper slice or if step is used
        """
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
            return pick_range(self, i, j)

    cpdef scalar_value(self, recalculate=False):
        """Returns value of an expression as a scalar
        
        This only works if the expression is a scalar
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: False)
        
        Returns:
            float: Scalar value of the expression
        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        if recalculate: self.cg().forward(self.vindex) # TODO: make recalculate run on the entire graph, not only up to here?
        return c_as_scalar(self.cgp().get_value(self.vindex))

    cpdef vec_value(self, recalculate=False):
        """Returns the value of the expression as a vector
        
        In case of a multidimensional expression, the values are flattened according to a column major ordering
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: False)
        
        Returns:
            list: Array of values
        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        if recalculate: self.cg().forward(self.vindex)
        return c_as_vector(self.cgp().get_value(self.vindex))

    cpdef npvalue(self, recalculate=False):
        """Returns the value of the expression as a numpy array
        
        The last dimension is the batch size (if it's > 1)
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: False)
        
        Returns:
            np.ndarray: numpy array of values
        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        cdef CTensor t
        cdef CDim dim
        if recalculate: self.cg().forward(self.vindex)
        t = self.cgp().get_value(self.vindex)
        dim = t.d
        arr = np.array(c_tensor_as_np(t))
        return arr

    cpdef tensor_value(self, recalculate=False):
        """Returns the value of the expression as a Tensor.

        This is useful if you want to use the value for other on-device calculations
        that are not part of the computation graph, i.e. using argmax.
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: False)
        
        Returns:
            Tensor: a dynet Tensor object.
        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        cdef CTensor t
        cdef CDim dim
        if recalculate: self.cg().forward(self.vindex)
        t = self.cgp().get_value(self.vindex)
        return Tensor.wrap_ctensor(t)

    cpdef value(self, recalculate=False):
        """Gets the value of the expression in the most relevant format
        
        this returns the same thing as :code:`scalar_value`, :code:`vec_value`, :code:`npvalue` depending on whether the number of dimensions of the expression is 0, 1 or 2+
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: False)
        
        Returns:
            float, list, np.ndarray: Value of the expression
        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        cdef CTensor t
        if recalculate: self.cg().forward(self.vindex)
        t = self.cgp().get_value(self.vindex)
        if t.d.ndims() >= 2:
            return self.npvalue()
        vec = self.vec_value()
        if len(vec) == 1: return vec[0]
        return vec

    cpdef gradient(self):
        """Returns the value of the expression as a numpy array
        
        The last dimension is the batch size (if it's > 1).

        Make sure to call :code:`backward` on a downstream expression before calling this.

        If the Expression is a constant expression (meaning it's not a function of a parameter), dynet won't compute it's gradient for the sake of efficiency. You need to manually force the gradient computation by adding the agument :code:`full=True` to :code:`backward`
        
        Returns:
            np.ndarray: numpy array of values
        """
        cdef CTensor t
        cdef CDim dim
        t = self.c().gradient()
        dim = t.d
        arr = c_tensor_as_np(t)
        return arr

    # TODO this runs incremental forward on the entire graph, may not be optimal in terms of efficiency.
    cpdef forward(self, recalculate=False):
        """This runs incremental forward on the entire graph
        
        May not be optimal in terms of efficiency.
        Prefer :code:`values`
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: False)
        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        if recalculate: self.cg().forward(self.vindex)
        else: self.cg().inc_forward(self.vindex)

    cpdef backward(self, bool full=False):
        """Run the backward pass based on this expression
        
        The parameter :code:`full` specifies whether the gradients should be computed for all nodes (:code:`True`) or only non-constant nodes (:code:`False`).
        
        By default, a node is constant unless
        
        1. it is a parameter node
        2. it depends on a non-constant node
        
        Thus, functions of constants and inputs are considered as constants.
        
        Turn :code:`full` on if you want to retrieve gradients w.r.t. inputs for instance. By default this is turned off, so that the backward pass ignores nodes which have no influence on gradients w.r.t. parameters for efficiency.

        Args:
            full (bool): Whether to compute all gradients (including with respect to constant nodes).

        """
        if self.cg_version != _cg._cg_version: raise RuntimeError("Stale Expression (created before renewing the Computation Graph).")
        self.cgp().backward(self.vindex, full)

    def __add__(self, other):
        if isinstance(self, Expression) and isinstance(other, Expression):
            return _add(self,other)
        elif isinstance(self, (int,float)):
            return _cadd(other, self)
        elif isinstance(other, (int,float)):
            return _cadd(self, other)
        else: raise NotImplementedError()
    def __mul__(self, other):
        if isinstance(self, Expression) and isinstance(other, Expression):
            return _mul(self,other)
        elif isinstance(self, (int,float)):
            return _cmul(other, self)
        elif isinstance(other, (int,float)):
            return _cmul(self, other)
        else: raise NotImplementedError()
    def __div__(self, other):
        if isinstance(self, Expression) and isinstance(other, (int,float)):
            return _cdiv(self, other)
        else: raise NotImplementedError()
    def __truediv__(self, other):
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

cpdef forward(list exps, recalculate=False):
    cdef Expression maxe = exps[0]
    cdef Expression e
    for e in exps:
        if e.vindex > maxe.vindex: maxe = e
    maxe.forward(recalculate)

cpdef npvalues(list exps, recalculate=False):
    cdef Expression e
    forward(exps, recalculate)
    return [e.npvalue() for e in exps]

cpdef values(list exps, recalculate=False):
    cdef Expression e
    forward(exps, recalculate)
    return [e.value() for e in exps]

#cdef Expression _parameter(ComputationGraph g, Parameters p):
#    return Expression.from_cexpr(g.version(), c_parameter(g.thisptr[0], p.thisptr))

def parameter(p, update=True):
    """Load a parameter in the computation graph
    
    Get the expression corresponding to a parameter
    
    Args:
        p(Parameter,LookupParameter): Parameter to load (can be a lookup parameter as well)
        update(bool): If this is set to False, the parameter won't be updated during the backward pass
    
    Returns:
        Expression: Parameter expression
    
    Raises:
        NotImplementedError: Only works with parameters and lookup parameters
    """
    if isinstance(p,Parameters) or isinstance(p,LookupParameters):
        return p.expr(update)
    else:
        raise NotImplementedError("Cannot call parameter() on anything other than Parameters or LookupParameters")

# {{{ Mutable Expressions
#     These depend values that can be set by the caller

cdef class _inputExpression(Expression):
    """Subclass of Expression corresponding to scalar input expressions
    
    """
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
        """Change the value of the expression
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            s(float): New value
        """
        self.cgp().invalidate()
        self.val.set(s)

def scalarInput(float s):
    return _cg.inputValue(s)

cdef class _vecInputExpression(Expression):
    """Subclass of Expression corresponding to any non-scalar input expressions
    
    Despite the name, this also represents tensors (in column major format).
    TODO : change this
    """
    cdef FloatVectorValue val
    def __cinit__(self, ComputationGraph g, vector[float] val, dim=None,batch_size=1):
        self.val = FloatVectorValue(val)
        if dim is None: dim = self.val.size()
        #self.cg = g.thisptr
        self.cg_version = g.version()
        cdef CExpression e
        e = c_input(self.cgp()[0], Dim(dim,batch_size=batch_size), self.val.addr())
        self.vindex = e.i
        g._inputs.append(self)
    def set(self, vector[float] data):
        """Change the value of the expression
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        For now this only accepts new values as flattened arrays (column majors). TODO : change this

        Args:
            data(vector[float]): New value
        """
        self.cgp().invalidate()
        self.val.set(data)

cdef class _sparseInputExpression(Expression):
    """Subclass of Expression corresponding to any non-scalar input expressions
    
    Despite the name, this also represents tensors (in column major format).
    TODO : change this
    """
    def __cinit__(self, ComputationGraph g, vector[unsigned] idxs, vector[float] val, dim ,batch_size=1, defval=0):
        #self.cg = g.thisptr
        self.cg_version = g.version()
        cdef CExpression e
        e = c_input(self.cgp()[0], Dim(dim, batch_size=batch_size), idxs, val, defval)
        self.vindex = e.i
        g._inputs.append(self)

    def set(self, vector[float] data):
        """Change the value of the expression
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        For now this only accepts new values as flattened arrays (column majors). TODO : change this

        Args:
            data(vector[float]): New value
        """
        raise ValueError('Can\'t set value of sparse input vector for now')

def vecInput(int dim):
    """Input an empty vector
    
    Args:
        dim(number): Size
    
    Returns:
        _vecInputExpression: Corresponding expression
    """
    return _cg.inputVector(dim)

def inputVector(vector[float] v):
    """Input a vector by values
    
    Args:
        v(vector[float]): Values
    
    Returns:
        _vecInputExpression: Corresponding expression
    """
    return _cg.inputVectorLiteral(v)

def matInput(int d1, int d2):
    """DEPRECATED : use inputTensor
    
    TODO : remove this
    
    Args:
        int d1([type]): [description]
        int d2([type]): [description]
    
    Returns:
        dynet.Expression: [description]
    """
    raise DeprecationWarning('matInput is now deprecated. Use dynet.inputTensor instead')

def inputMatrix(vector[float] v, tuple d):
    """DEPRECATED : use inputTensor

    TODO : remove this

    inputMatrix(vector[float] v, tuple d)

    Create a matrix literal.
    First argument is a list of floats (or a flat numpy array).
    Second argument is a dimension.
    Returns: an expression.
    Usage example::

        x = inputMatrix([1,2,3,4,5,6],(2,3))
        x.npvalue()
        --> 
        array([[ 1.,  3.,  5.],
               [ 2.,  4.,  6.]])
    """
    raise DeprecationWarning('matInput is now deprecated. Use dynet.inputTensor instead')

def inputTensor(arr,batched=False):
    """Creates a tensor expression based on a numpy array or a list.
    
    The dimension is inferred from the shape of the input.
    if batched=True, the last dimension is used as a batch dimension
    if arr is a list of numpy ndarrays, this returns a batched expression where the batch elements are the elements of the list
    
    Args:
        arr(list,np.ndarray): Values : numpy ndarray OR list of np.ndarray OR multidimensional list of floats
    
    Keyword Args:
        batched(bool): Whether to use the last dimension as a batch dimension (default: False)
    
    Returns:
        _vecInputExpression: Input expression
    
    Raises:
        TypeError: If the type is not respected
    """
    if isinstance(arr,list):
        if all([isinstance(x,np.ndarray) for x in arr]):
            arr = np.stack(arr,axis=-1)
            batched=True
        else:
            arr=np.asarray(arr,dtype=float)
    if not isinstance(arr,np.ndarray):
        raise TypeError("Input Tensor should be a numpy.ndarray or a valid list of floats")
    if batched:
        dim = arr.shape[:-1] if len(arr.shape) > 1 else (1,)
        batch_size= arr.shape[-1]
    else:
        dim = arr.shape
        batch_size= 1
    arr = arr.flatten(order='F')
    return _cg.inputMatrixLiteral(arr, dim,batch_size=batch_size)


def sparse_inputTensor(idxs, values, shape, batched=False, defval=0):
    """Creates a tensor expression based on indices and values
    
    The dimension is inferred from the shape of the input.
    if batched=True, the last dimension is used as a batch dimension
    if arr is a list of numpy ndarrays, this returns a batched expression where the batch elements are the elements of the list
    
    Args:
        idxs(tuple, list): A tuple/list of integer arrays, one array for each dimension (including the batch dimension)
        values(list,np.ndarray): A 1D array/list of values
        shape: The desired shape
    Keyword Args:
        batched(bool): Whether to use the last dimension as a batch dimension (default: False). For example if :code:`shape=(3, 3, 3)` and :code:`batched=True` the resulting expression will be a batch of 3 3x3 matrices
        defval(number): The default value for all non specified coordinates (default: 0)
    
    Returns:
        _vecInputExpression: Input expression
    
    Raises:
        TypeError: If the type is not respected
        ValueError: If the number of dimensions don't match
    """
    if isinstance(values, list):
        values = np.asarray(values, dtype=float)
    if not len(values.shape) == 1:
        raise TypeError("values should be a 1d array")
    if not len(idxs) == len(shape):
        raise ValueError("Number of indices doesn't match shape")
    if batched:
        dim = shape[:-1] if len(shape) > 1 else (1,)
        batch_size= shape[-1]
    else:
        dim = shape
        batch_size = 1
    idxs = np.ravel_multi_index(idxs, shape, order='F')
    return _cg.inputSparseTensor(idxs, values, dim, batch_size=batch_size, defval=defval)

cdef class _lookupExpression(Expression):
    """Expression corresponding to a lookup from lookup parameter
    
    """
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
        """Change the lookup index
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(number): New lookup index
        """
        self.cgp().invalidate()
        self.val.set(i)

cdef class _lookupBatchExpression(Expression):
    """Expression corresponding to batched lookups from a lookup parameter
    
    """
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
        """Change the lookup index
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(list(int)): New indices
        """
        self.cgp().invalidate()
        self.val.set(i)

def lookup(LookupParameters p, unsigned index=0, update=True):
    """Pick an embedding from a lookup parameter and returns it as a expression

        :param p: Lookup parameter to pick from
        :type p: LookupParameters
    
    Keyword Args:
        index(number): Lookup index (default: 0)
        update(bool): Whether to update the lookup parameter [(default: True)
    
    Returns:
        _lookupExpression: Expression for the embedding
    """
    return _cg.lookup(p, index, update)

def lookup_batch(LookupParameters p, vector[unsigned] indices, update=True):
    """Look up parameters.

    The mini-batched version of lookup. The resulting expression will be a mini-batch of parameters, where the "i"th element of the batch corresponds to the parameters at the position specified by the "i"th element of "indices"
    
    Args:
        p(LookupParameters): Lookup parameter to pick from
        indices(list(int)): Indices to look up for each batch element
    
    Keyword Args:
        update(bool): Whether to update the lookup parameter (default: True)
    
    Returns:
        _lookupBatchExpression: Expression for the batched embeddings
    """
    return _cg.lookup_batch(p, indices, update)

cdef class _pickerExpression(Expression):
    """Expression corresponding to a row picked from a bigger expression
    
    """
    cdef UnsignedValue val
    cdef unsigned dim
    def __cinit__(self, ComputationGraph g, Expression e, unsigned index=0, unsigned dim=0):
        self.val = UnsignedValue(index)
        self.dim = dim
        #self.cg = e.cg
        self.cg_version = g.version()
        cdef CExpression ce
        ce = c_pick(e.c(), self.val.addr(), self.dim)
        self.vindex = ce.i
        g._inputs.append(self)
    def set_index(self,i):
        """Change the pick index
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(number): New index
        """
        self.cgp().invalidate()
        self.val.set(i)

def pick(Expression e, unsigned index=0, unsigned dim=0):
    """Pick element.

    Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.
    
    Args:
        e(Expression): Expression to pick from
    
    Keyword Args:
        index(number): Index to pick (default: 0)
        dim(number): Dimension to pick from (default: 0)
    
    Returns:
        _pickerExpression: Picked expression
    """
    return _cg.outputPicker(e, index, dim)

cdef class _pickerBatchExpression(Expression):
    """Batched version of :code:`_pickerExpression`
    
    """
    cdef UnsignedVectorValue val
    cdef unsigned dim
    def __cinit__(self, ComputationGraph g, Expression e, vector[unsigned] indices, unsigned dim=0):#
        self.val = UnsignedVectorValue(indices)
        self.dim = dim
        self.cg_version = g.version()
        cdef CExpression ce
        ce = c_pick(e.c(), self.val.addr(), self.dim)
        self.vindex = ce.i
        g._inputs.append(self)
    def set_index(self,i):
        """Change the pick indices
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use :code::code:`recalculate=True` when calling :code:`.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(list): New list of indices
        """
        self.cgp().invalidate()
        self.val.set(i)

def pick_batch(Expression e, vector[unsigned] indices, unsigned dim=0):
    """Batched pick.

    Pick elements from multiple batches.
    
    Args:
        e(Expression): Expression to pick from
        indices(list): Indices to pick
        dim(number): Dimension to pick from (default: 0)
    
    Returns:
        _pickerBatchExpression: Picked expression
    """
    return _cg.outputBatchPicker(e, indices, dim)

# }}}

cpdef Expression zeroes(dim, int batch_size=1): 
    """Create an input full of zeros
    
    Create an input full of zeros, sized according to dimensions :code:`dim`
    
    Args:
        dim (tuple): Dimension of the tensor
    
    Keyword Arguments:
        batch_size (number): Batch size of the tensor (default: (1))
    
    Returns:
        dynet.Expression: A "d" dimensioned zero tensor
    """
    return Expression.from_cexpr(_cg.version(), c_zeroes(_cg.thisptr[0], CDim(dim, batch_size)))
cpdef Expression random_normal(dim, int batch_size=1): 
    """Create a random normal vector
    
    Create a vector distributed according to normal distribution with mean 0, variance 1.
    
    Args:
        dim (tuple): Dimension of the tensor
    
    Keyword Arguments:
        batch_size (number): Batch size of the tensor  (default: (1))
    
    Returns:
        dynet.Expression: A "d" dimensioned normally distributed tensor
    """
    return Expression.from_cexpr(_cg.version(), c_random_normal(_cg.thisptr[0], CDim(dim, batch_size)))
cpdef Expression random_bernoulli(dim, float p, float scale=1.0, int batch_size=1):
    """Create a random bernoulli tensor
    
    Create a tensor distributed according to bernoulli distribution with parameter :math:`p`.
    
    Args:
        dim (tuple): Dimension of the tensor
        p (number): Parameter of the bernoulli distribution
    
    Keyword Arguments:
        scale (number): Scaling factor to apply to the sampled tensor (default: (1.0))
        batch_size (number): Batch size of the tensor (default: (1))
    
    Returns:
        dynet.Expression: A "d" dimensioned bernoulli distributed tensor
    """
    return Expression.from_cexpr(_cg.version(), c_random_bernoulli(_cg.thisptr[0], CDim(dim, batch_size), p, scale))
cpdef Expression random_uniform(dim, float left, float right, int batch_size=1):
    """Create a random uniform tensor
    
    Create a tensor distributed according to uniform distribution with boundaries left and right.

    Args:
        dim (tuple): Dimension of the tensor
        left (number): Lower bound of the uniform distribution
        right (number): Upper bound of the uniform distribution
    
    Keyword Arguments:
        batch_size (number): Batch size of the tensor (default: (1))
    
    Returns:
        dynet.Expression: A "d" dimensioned uniform distributed tensor
    """
    return Expression.from_cexpr(_cg.version(), c_random_uniform(_cg.thisptr[0], CDim(dim, batch_size), left, right))
cpdef Expression random_gumbel(dim, float mu = 0.0, float beta = 1.0, int batch_size=1):
    """Create a random Gumbel sampled vector
    
    Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported.
    
    Args:
        dim (tuple): Dimension of the tensor
    
    Keyword Arguments:
        mu (number): The :math:`\mu` parameter (default: (0.0))
        beta (number): The :math:`\\beta` parameter (default: (1.0))
        batch_size (number): Batch size of the tensor (default: (1))
    
    Returns:
        dynet.Expression:  "d" dimensioned Gumbel distributed tensor
    """
    return Expression.from_cexpr(_cg.version(), c_random_gumbel(_cg.thisptr[0], CDim(dim, batch_size), mu, beta))

cpdef Expression nobackprop(Expression x):
    """Prevent backprop
    
    This node has no effect on the forward pass, but prevents gradients from flowing backward during the backward pass. This is useful when there's a subgraph for which you don't want loss passed back to the parameters.
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: An output expression containing the same as input (only effects on backprop process)
    """
    return Expression.from_cexpr(x.cg_version, c_nobackprop(x.c()))
cpdef Expression flip_gradient(Expression x):
    """Negative backprop
    
    This node has no effect on the forward pass, but takes negative on backprop process. This operation is widely used in adversarial networks.
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: An output expression containing the same as input (only effects on backprop process)
    """
    return Expression.from_cexpr(x.cg_version, c_flip_gradient(x.c()))

# binary-exp
cpdef Expression cdiv(Expression x, Expression y):
    """Componentwise division
    
    Do a componentwise division where each value is equal to :math:`\\frac{x_i}{y_i}`
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression

    Returns:
        dynet.Expression: An expression where the ith element is equal to :math:`\\frac{x_i}{y_i}`
    """
    ensure_freshness(y);
    return Expression.from_cexpr(x.cg_version, c_cdiv(x.c(), y.c()))
cpdef Expression cmult(Expression x, Expression y):
    """Componentwise multiplication
    
    Do a componentwise multiplication where each value is equal to :math:`x_i\\times y_i`
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression

    Returns:
        dynet.Expression: An expression where the ith element is equal to :math:`x_i\\times y_i`
    """
    ensure_freshness(y);
    return Expression.from_cexpr(x.cg_version, c_cmult(x.c(), y.c()))
cpdef Expression colwise_add(Expression x, Expression y):
    """Columnwise addition
    
    Add vector :math:`y` to each column of matrix :math:`x`
    
    Args:
        x (dynet.Expression): An MxN matrix
        y (dynet.Expression): A length M vector
    
    Returns:
        dynet.Expression: An expression where :math:`y` is added to each column of :math:`x`
    """
    ensure_freshness(y);
    return Expression.from_cexpr(x.cg_version, c_colwise_add(x.c(), y.c()))

cpdef Expression inverse(Expression x): 
    """Matrix Inverse
    
    Takes the inverse of a matrix (not implemented on GPU yet, although contributions are welcome: `issue <https://github.com/clab/dynet/issues/158>`_). Note that back-propagating through an inverted matrix can also be the source of stability problems sometimes.
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: Inverse of x
    """
    return Expression.from_cexpr(x.cg_version, c_inverse(x.c()))
cpdef Expression logdet(Expression x): 
    """Log determinant

    Takes the log of the determinant of a matrix. (not implemented on GPU yet, although contributions are welcome: `issue <https://github.com/clab/dynet/issues/158>`_).
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`\log(\\vert x\\vert)`
    """
    return Expression.from_cexpr(x.cg_version, c_logdet(x.c()))
cpdef Expression trace_of_product(Expression x, Expression y):
    """Trace of Matrix Product

    Takes the trace of the product of matrices. (not implemented on GPU yet, although contributions are welcome: `issue <https://github.com/clab/dynet/issues/158>`_).
    
    Args:
        x (dynet.Expression): The first input expression
        Expression y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`\\text{Tr}(xy)`
    """
    ensure_freshness(y);
    return Expression.from_cexpr(x.cg_version, c_trace_of_product(x.c(), y.c()))
cpdef Expression dot_product(Expression x, Expression y):
    """Dot Product
    
    Calculate the dot product :math:`x^Ty=\sum_i x_iy_i`
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`x^Ty=\sum_i x_iy_i`
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_dot_product(x.c(), y.c()))
cpdef Expression squared_norm(Expression x):
    """Squared norm
    
    The squared norm of the values of :code:`x`: :math:`\Vert x\Vert_2^2=\sum_i x_i^2`.

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`\Vert x\Vert_2^2=\sum_i x_i^2`
    """
    return Expression.from_cexpr(x.cg_version, c_squared_norm(x.c()))
cpdef Expression l2_norm(Expression x):
    """L2 norm

    The l2 norm of the values of :code:`x`: :math:`\Vert x\Vert_2=\sqrt{\sum_i x_i^2}`.

    Args:
        x (dynet.Expression): Input expression

    Returns:
        dynet.Expression: :math:`\Vert x\Vert_2=\sqrt{\sum_i x_i^2}`
    """
    return Expression.from_cexpr(x.cg_version, c_l2_norm(x.c()))
cpdef Expression squared_distance(Expression x, Expression y):
    """Squared distance
    
    The squared distance between values of :code:`x` and :code:`y`: :math:`\Vert x-y\Vert_2^2=\sum_i (x_i-y_i)^2`.
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`\Vert x-y\Vert_2^2=\sum_i (x_i-y_i)^2`
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_squared_distance(x.c(), y.c()))
cpdef Expression l1_distance(Expression x, Expression y):
    """L1 distance
    
    L1 distance between values of :code:`x` and :code:`y`: :math:`\Vert x-y\Vert_1=\sum_i \\vert x_i-y_i\\vert`.
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`\Vert x-y\Vert_1=\sum_i \\vert x_i-y_i\\vert`.
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_l1_distance(x.c(), y.c()))
cpdef Expression binary_log_loss(Expression x, Expression y):
    """Binary log loss
    
    The log loss of a binary decision according to the sigmoid sigmoid function :math:`- \sum_i (y_i  \ln(x_i) + (1-y_i)  \ln(1-x_i))`
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`- \sum_i (y_i  \ln(x_i) + (1-y_i)  \ln(1-x_i))`
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_binary_log_loss(x.c(), y.c()))
#cpdef Expression conv1d_narrow(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_conv1d_narrow(x.c(), y.c()))
#cpdef Expression conv1d_wide(Expression x, Expression y): ensure_freshness(y); return Expression.from_cexpr(x.cg_version, c_conv1d_wide(x.c(), y.c()))
cpdef Expression filter1d_narrow(Expression x, Expression y): 
    """[summary]
    
    [description]
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression

    Returns:
        dynet.Expression: TODO
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_filter1d_narrow(x.c(), y.c()))
cpdef Expression conv2d(Expression x, Expression f, vector[unsigned] stride, bool is_valid = True): 
    """2D convolution without bias
    
    2D convolution operator without bias parameters.
    :code:`VALID` and :code:`SAME` convolutions are supported.
    
    Think about when stride is 1, the distinction:
    
    - :code:`SAME`: output size is the same with input size. To do so, one needs to pad the input so the filter can sweep outside of the input maps.
    - :code:`VALID`: output size shrinks by :code:`filter_size - 1`, and the filters always sweep at valid positions inside the input maps. No padding needed.

    In detail, assume
    
    - Input feature maps: :code:`XH x XW x XC x N`
    - Filters: :code:`FH x FW x XC x FC`
    - Strides: :code:`strides[0]` and :code:`strides[1]` are row (:code:`h`) and col (:code:`w`) stride, respectively.
 
    For the :code:`SAME` convolution: the output height (:code:`YH`) and width (:code:`YW`) are computed as:
    
    - :code:`YH = ceil(float(XH) / float(strides[0]))`
    - :code:`YW = ceil(float(XW) / float(strides[1]))`
    
    and the paddings are computed as:
    
    - :code:`pad_along_height = max((YH - 1) * strides[0] + FH - XH, 0)`
    - :code:`pad_along_width = max((YW - 1) * strides[1] + FW - XW, 0)`
    - :code:`pad_top = pad_along_height / 2`
    - :code:`pad_bottom = pad_along_height - pad_top`
    - :code:`pad_left = pad_along_width / 2`
    - :code:`pad_right = pad_along_width - pad_left`
 
    For the :code:`VALID` convolution: the output height (:code`YH`) and width (:code:`YW`) are computed as:
    
    - :code:`YH = ceil(float(XH - FH + 1) / float(strides[0]))`
    - :code:`YW = ceil(float(XW - FW + 1) / float(strides[1]))`
    
    and the paddings are always zeros.
    
    Args:
        x (dynet.Expression): The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with an optional batch dimension
        f (dynet.Expression): 2D convolution filters: H x W x Ci x Co (ColMaj), 4D tensor
        stride (list): the row and column strides
    
    Keyword Arguments:
        is_valid (bool): 'VALID' convolution or 'SAME' convolution, default is True ('VALID') (default: (True))
    
    Returns:
        dynet.Expression: The output feature maps (H x W x Co) x N, 3D tensor with an optional batch dimension
    """
    ensure_freshness(f); 
    return Expression.from_cexpr(x.cg_version, c_conv2d(x.c(), f.c(), stride, is_valid))

cpdef Expression conv2d_bias(Expression x, Expression f, Expression b, vector[unsigned] stride, bool is_valid = True):
    """2D convolution with bias
    
    2D convolution operator with bias parameters.
    :code:`VALID` and :code:`SAME` convolutions are supported.
    
    Think about when stride is 1, the distinction:
    
    - :code:`SAME`: output size is the same with input size. To do so, one needs to pad the input so the filter can sweep outside of the input maps.
    - :code:`VALID`: output size shrinks by :code:`filter_size - 1`, and the filters always sweep at valid positions inside the input maps. No padding needed.

    In detail, assume
    
    - Input feature maps: :code:`XH x XW x XC x N`
    - Filters: :code:`FH x FW x XC x FC`
    - Strides: :code:`strides[0]` and :code:`strides[1]` are row (:code:`h`) and col (:code:`w`) stride, respectively.
 
    For the :code:`SAME` convolution: the output height (:code:`YH`) and width (:code:`YW`) are computed as:
    
    - :code:`YH = ceil(float(XH) / float(strides[0]))`
    - :code:`YW = ceil(float(XW) / float(strides[1]))`
    
    and the paddings are computed as:
    
    - :code:`pad_along_height = max((YH - 1) * strides[0] + FH - XH, 0)`
    - :code:`pad_along_width = max((YW - 1) * strides[1] + FW - XW, 0)`
    - :code:`pad_top = pad_along_height / 2`
    - :code:`pad_bottom = pad_along_height - pad_top`
    - :code:`pad_left = pad_along_width / 2`
    - :code:`pad_right = pad_along_width - pad_left`
 
    For the :code:`VALID` convolution: the output height (:code`YH`) and width (:code:`YW`) are computed as:
    
    - :code:`YH = ceil(float(XH - FH + 1) / float(strides[0]))`
    - :code:`YW = ceil(float(XW - FW + 1) / float(strides[1]))`
    
    and the paddings are always zeros.
    
    Args:
        x (dynet.Expression): The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with an optional batch dimension
        f (dynet.Expression): 2D convolution filters: H x W x Ci x Co (ColMaj), 4D tensor
        b (dynet.Expression): The bias (1D: Ci)
        stride (list): the row and column strides
    
    Keyword Arguments:
        is_valid (bool): 'VALID' convolution or 'SAME' convolution, default is True ('VALID') (default: (True))
    
    Returns:
        dynet.Expression: The output feature maps (H x W x Co) x N, 3D tensor with an optional batch dimension
    """
    ensure_freshness(f)
    ensure_freshness(b)
    return Expression.from_cexpr(x.cg_version, c_conv2d(x.c(), f.c(), b.c(), stride, is_valid))

cpdef Expression maxpooling2d(Expression x, vector[unsigned] ksize, vector[unsigned] stride, bool is_valid = True): 
    """2D maxpooling
    
    2D maxpooling operator.
    :code:`VALID` and :code:`SAME` maxpooling are supported.
    
    Args:
        x (dynet.Expression): The input feature maps: (H x W x Ci) x N (ColMaj), 3D tensor with an optional batch dimension
        ksize (list): the max pooling 2d window size 
        stride (list): the row and column strides
    
    Keyword Arguments:
        is_valid (bool): 'VALID' or 'SAME', default is True ('VALID') (default: (True))
    
    Returns:
        dynet.Expression: The output feature maps (H x W x Co) x N, 3D tensor with an optional batch dimension
    """
    return Expression.from_cexpr(x.cg_version, c_maxpooling2d(x.c(), ksize, stride, is_valid))

# unary-exp
cpdef Expression tanh(Expression x): 
    """Hyperbolic tangent
    
    Elementwise calculation of the hyperbolic tangent
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`\\tanh(x)`
    """
    return Expression.from_cexpr(x.cg_version, c_tanh(x.c()))
cpdef Expression exp(Expression x): 
    """Natural exponent
    
    Calculate elementwise :math:`y_i = e^{x_i}`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`e^{x}`
    """
    return Expression.from_cexpr(x.cg_version, c_exp(x.c()))
cpdef Expression square(Expression x): 
    """Square
    
    Calculate elementwise :math:`y_i = x_i^2`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y = x^2`
    """
    return Expression.from_cexpr(x.cg_version, c_square(x.c()))
cpdef Expression sqrt(Expression x): 
    """Square root
    
    Calculate elementwise :math:`y_i = \sqrt{x_i}`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y = \sqrt{x}`
    """
    return Expression.from_cexpr(x.cg_version, c_sqrt(x.c()))

cpdef Expression abs(Expression x): 
    """Absolute value
    
    Calculate elementwise :math:`y_i = \\vert x_i\\vert`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y = \\vert x\\vert`
    """
    return Expression.from_cexpr(x.cg_version, c_abs(x.c()))

cpdef Expression erf(Expression x): 
    """Gaussian error function

    Elementwise calculation of the Gaussian error function :math:`y_i = \\text{erf}(x_i)=\\frac {1}{\sqrt{\pi}}\int_{-x_i}^{x_i}e^{-t^2}\mathrm{d}t`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y_i = \\text{erf}(x_i)`
    """
    return Expression.from_cexpr(x.cg_version, c_erf(x.c()))
cpdef Expression cube(Expression x): 
    """cube
    
    Calculate elementwise :math:`y_i = x_i^3`

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y = x^3`
    """
    return Expression.from_cexpr(x.cg_version, c_cube(x.c()))
cpdef Expression log(Expression x): 
    """Natural logarithm
    
    Elementwise calculation of the natural logarithm :math:`y_i = \ln(x_i)`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y_i = \ln(x_i)`
    """
    return Expression.from_cexpr(x.cg_version, c_log(x.c()))
cpdef Expression lgamma(Expression x): 
    """Log gamma
    
    Calculate elementwise log gamma function :math:`y_i = \ln(\Gamma(x_i))`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y_i = \ln(\Gamma(x_i))`
    """
    return Expression.from_cexpr(x.cg_version, c_lgamma(x.c()))
cpdef Expression logistic(Expression x): 
    """Logistic sigmoid function
    
    Calculate elementwise :math:`y_i = \\frac{1}{1+e^{-x_i}}`
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y_i = \\frac{1}{1+e^{-x_i}}`
    """
    return Expression.from_cexpr(x.cg_version, c_logistic(x.c()))

cpdef Expression rectify(Expression x): 
    """Rectifier (or ReLU, Rectified Linear Unit)
    
    Calculate elementwise recitifer (ReLU) function :math:`y_i = \max(x_i,0)`

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y_i = \max(x_i,0)`
    """
    return Expression.from_cexpr(x.cg_version, c_rectify(x.c()))

cpdef Expression elu(Expression x, float alpha=1.0): 
    """Exponential Linear Unit (ELU)

    Calculate elementwise the function 

    .. math::
        y_i = \left\{\\begin{array}{lr}
                   x_i, & \\text{if } x>0\\\\
                   \\alpha\\times(e^{x_i} - 1), & \\text{if }x\leqslant 0
                 \end{array}\\right.
        
    Reference: `Clevert et al., 2015 <https://arxiv.org/abs/1511.07289v5>`_
 
    Args:
        x (dynet.Expression): Input expression
        alpha (number): :math:`\\alpha` parameter
    
    Returns:
        dynet.Expression: :math:`\\text{ELU}(x_i, \\alpha)`
    """
    return Expression.from_cexpr(x.cg_version, c_elu(x.c(), alpha))

cpdef Expression selu(Expression x): 
    """Scaled Exponential Linear Unit (SELU)

    Calculate elementwise the function 

    .. math::
        y_i = \lambda\\times\left\{
        \\begin{array}{lr}
           x_i, & \\text{if } x>0\\\\
           \\alpha\\times(e^{x_i} - 1), & \\text{if }x\leqslant 0\\\\
        \end{array}\\right.

    With

    .. math::
        \\begin{split}
            \lambda &=\\texttt{1.0507009873554804934193349852946}\\\\
            \\alpha &=\\texttt{1.6732632423543772848170429916717}\\\\
        \end{split}

    Reference: `Klambaouer et al., 2017 <https://arxiv.org/abs/1706.02515>`_
 
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`\\text{SELU}(x_i)`
    """
    return Expression.from_cexpr(x.cg_version, c_selu(x.c()))

cpdef Expression log_softmax(Expression x, list restrict=None):
    """Restricted log softmax
    
    The log softmax function calculated over only a subset of the vector elements. The elements to be included are set by the :code:`restriction` variable. All elements not included in :code:`restriction` are set to negative infinity.
    
    Args:
        x (dynet.Expression): Input expression
    
    Keyword Arguments:
        restrict (list): List of log softmax to compute (default: (None))
    
    Returns:
        dynet.Expression: A vector with the log softmax over the specified elements
    """
    if restrict is None:
        return Expression.from_cexpr(x.cg_version, c_log_softmax(x.c()))
    cdef vector[unsigned] vec = restrict
    return Expression.from_cexpr(x.cg_version, c_log_softmax(x.c(), vec))
cpdef Expression softmax(Expression x):
    """Softmax
    
    The softmax function normalizes each column to ensure that all values are between 0 and 1 and add to one by applying the :math:`\\frac{e^{x_i}}{sum_j e^{x_j}}`.
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`\\frac{e^{x_i}}{\sum_j e^{x_j}}`
    """
    return Expression.from_cexpr(x.cg_version, c_softmax(x.c()))
cpdef Expression sparsemax(Expression x):
    """Sparsemax
    
    The sparsemax function (Martins et al. 2016), which is similar to softmax, but induces sparse solutions where most of the vector elements are zero. **Note:** This function is not yet implemented on GPU.
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: The sparsemax of the scores
    """
    return Expression.from_cexpr(x.cg_version, c_sparsemax(x.c()))
cpdef Expression softsign(Expression x):
    """Softsign function

    Calculate elementwise the softsign function :math:`y_i = \\frac{x_i}{1+\\vert x_i\\vert}`

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: :math:`y_i = \\frac{x_i}{1+\\vert x_i\\vert}`
    """
    return Expression.from_cexpr(x.cg_version, c_softsign(x.c()))
cpdef Expression pow(Expression x, Expression y):
    """Power function
    
    Calculate an output where the ith element is equal to :math:`x_i^{y_i}`

    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`x_i^{y_i}`
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_pow(x.c(), y.c()))
cpdef Expression bmin(Expression x, Expression y):
    """Minimum
    
    Calculate an output where the ith element is :math:`\min(x_i,y_i)`

    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`\min(x_i,y_i)`
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_bmin(x.c(), y.c()))
cpdef Expression bmax(Expression x, Expression y):
    """Maximum
    
    Calculate an output where the ith element is :math:`\max(x_i,y_i)`

    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: :math:`\max(x_i,y_i)`
    """
    ensure_freshness(y); 
    return Expression.from_cexpr(x.cg_version, c_bmax(x.c(), y.c()))
cpdef Expression transpose(Expression x, list dims=[1, 0]):
    """Transpose a matrix
    
    Get the transpose of the matrix, or if dims is specified shuffle the dimensions arbitrarily.

    **Note:** This is O(1) if either the row or column dimension is 1, and O(n) otherwise.
    
    Args:
        x (dynet.Expression): Input expression
        dims (list): The dimensions to swap. The ith dimension of the output will be equal to the dims[i] dimension of the input. dims must have the same number of dimensions as x.
    
    Returns:
        dynet.Expression: :math:`x^T` / the shuffled expression
    """
    cdef vector[unsigned] vec = dims
    return Expression.from_cexpr(x.cg_version, c_transpose(x.c(), vec))
cpdef Expression select_rows(Expression x, vector[unsigned] rs):
    """Select rows

    Select a subset of rows of a matrix.

    Args:
        x (dynet.Expression): Input expression
        rs (list): The rows to extract 
    
    Returns:
        dynet.Expression: An expression containing the selected rows
    """
    return Expression.from_cexpr(x.cg_version, c_select_rows(x.c(), rs))
cpdef Expression select_cols(Expression x, vector[unsigned] cs):
    """Select columns

    Select a subset of columns of a matrix.

    Args:
        x (dynet.Expression): Input expression
        cs (list): The columns to extract 
    
    Returns:
        dynet.Expression: An expression containing the selected columns
    """
    return Expression.from_cexpr(x.cg_version, c_select_cols(x.c(), cs))
cpdef Expression sum_cols(Expression x):
    """[summary]
    
    [description]
    
    Args:
        x (dynet.Expression): 
    
    Returns:
        dynet.Expression: 
    """
    return Expression.from_cexpr(x.cg_version, c_sum_cols(x.c()))
cpdef Expression sum_elems(Expression x):
    """Sum all elements
    
    Sum all the elements in an expression.
    
    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: The sum of all of its elements
    """
    return Expression.from_cexpr(x.cg_version, c_sum_elems(x.c()))

cpdef Expression sum_batches(Expression x):
    """Sum over minibatches
    
    Sum an expression that consists of multiple minibatches into one of equal dimension but with only a single minibatch. This is useful for summing loss functions at the end of minibatch training.

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: An expression with a single batch
    """
    return Expression.from_cexpr(x.cg_version, c_sum_batches(x.c()))

cpdef Expression mean_elems(Expression x):
    """Mean of elements of the tensor
    
    Computes the mean :math:`\\frac 1 n \sum_ix_i` of all the elements of each minibatch.

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: A scalar expression (minibatched)
    """
    return Expression.from_cexpr(x.cg_version, c_mean_elems(x.c()))

cpdef Expression mean_batches(Expression x):
    """Mean along the batch dimension
    
    Computes the mean :math:`\\frac 1 n \sum_ix_i`  along the batch dimension.


    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: An expression with a single batch
    """
    return Expression.from_cexpr(x.cg_version, c_mean_batches(x.c()))

cpdef Expression std_elems(Expression x):
    """Standard deviation of elements of the tensor
    
    Computes the standard deviation :math:`\sigma=\sqrt{\\frac 1 n \sum_i(x_i-\mu)^2}` of all the elements of each minibatch.

    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: A scalar expression (minibatched)
    """
    return Expression.from_cexpr(x.cg_version, c_std_elems(x.c()))

cpdef Expression std_batches(Expression x):
    """Standard deviation along the batch dimension
    
    Computes the standard deviation :math:`\sigma=\sqrt{\\frac 1 n \sum_i(x_i-\mu)^2}`  along the batch dimension.


    Args:
        x (dynet.Expression): Input expression
    
    Returns:
        dynet.Expression: An expression with a single batch
    """
    return Expression.from_cexpr(x.cg_version, c_std_batches(x.c()))

cpdef Expression std_dim(Expression x, unsigned d):
    """Standard deviation along an arbitrary dimension
    
    Computes the standard deviation :math:`\sigma=\sqrt{\\frac 1 n \sum_i(x_i-\mu)^2}` along an arbitrary dimension.


    Args:
        x (dynet.Expression): Input expression
        d (int): Dimension along which to reduce
    
    Returns:
        dynet.Expression: An expression with one less dimension
    """
    return Expression.from_cexpr(x.cg_version, c_std_dim(x.c(), d))


cpdef Expression mean_dim(Expression x, unsigned d):
    """Mean along an arbitrary dimension
    
    Computes the mean :math:`\\frac 1 n \sum_ix_i`  along an arbitrary dimension.


    Args:
        x (dynet.Expression): Input expression
        d (int): Dimension along which to reduce
    
    Returns:
        dynet.Expression: An expression with one less dimension
    """
    return Expression.from_cexpr(x.cg_version, c_mean_dim(x.c(), d))

cpdef Expression moment_elems(Expression x, unsigned r):
    """Statistical moment of elements of the tensor
    
    Computes the statistical moment of order :math:`r`, :math:`\\frac 1 n \sum_ix_i^r` of all the elements of each minibatch.

    Args:
        x (dynet.Expression): Input expression
        r (int): Moment order
    
    Returns:
        dynet.Expression: A scalar expression (minibatched)
    """
    return Expression.from_cexpr(x.cg_version, c_moment_elems(x.c(), r))

cpdef Expression moment_batches(Expression x, unsigned r):
    """Statistical moment along the batch dimension
    
    Computes the statistical moment of order :math:`r`, :math:`\\frac 1 n \sum_ix_i^r`  along the batch dimension.


    Args:
        x (dynet.Expression): Input expression
        r (int): Moment order
    
    Returns:
        dynet.Expression: An expression with a single batch
    """
    return Expression.from_cexpr(x.cg_version, c_moment_batches(x.c(), r))

cpdef Expression moment_dim(Expression x, unsigned d, unsigned r):
    """Statistical moment along an arbitrary dimension
    
    Computes the statistical moment of order :math:`r`, :math:`\\frac 1 n \sum_ix_i^r`  along an arbitrary dimension.


    Args:
        x (dynet.Expression): Input expression
        d (int): Dimension along which to reduce
        r (int): Moment order
    
    Returns:
        dynet.Expression: An expression with one less dimension
    """
    return Expression.from_cexpr(x.cg_version, c_moment_dim(x.c(), d, r))

#expr-opt
cpdef Expression fold_rows(Expression x, unsigned nrows=2):
    """[summary]
    
    [description]
    
    Args:
        x (dynet.Expression): 
    
    Keyword Arguments:
        unsigned nrows {number}:  (default: (2))
    
    Returns:
        dynet.Expression: 
    """
    return Expression.from_cexpr(x.cg_version, c_fold_rows(x.c(),nrows))
#expr-expr-opt
# x is scalar or row vector
# y is scalar or row vector
# res = max(0, m - x + y)
cpdef Expression pairwise_rank_loss(Expression x, Expression y, float m=1.0):
    """Pairwise rank loss
    
    A margin-based loss, where every margin violation for each pair of values is penalized: :math:`\sum_i \max(x_i-y_i+m, 0)`
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Keyword Arguments:
        m (number): The margin (default: (1.0))
    
    Returns:
        dynet.Expression: The pairwise rank loss
    """
    ensure_freshness(y);
    return Expression.from_cexpr(x.cg_version, c_pairwise_rank_loss(x.c(), y.c(), m))
cpdef Expression poisson_loss(Expression x, unsigned y):
    """Poisson loss
    
    The negative log probability of :code:`y` according to a Poisson distribution with parameter :code:`x`. Useful in Poisson regression where, we try to predict the parameters of a Possion distribution to maximize the probability of data :code:`y`.
    
    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Returns:
        dynet.Expression: The Poisson loss
    """
    return Expression.from_cexpr(x.cg_version, c_poisson_loss(x.c(), y))
cpdef Expression huber_distance(Expression x, Expression y, float c=1.345):
    """Huber distance
    
    The huber distance between values of :code:`x` and :code:`y` parameterized by :code:`c`, :math:`\sum_i L_c(x_i, y_i)` where:

    .. math::

        L_c(x, y) = \\begin{cases}{lr}
        \\frac{1}{2}(y - x)^2                   & \\textrm{for } \\vert y - f(x)\\vert  \le c, \\\\
        c\, \\vert y - f(x)\\vert  - \\frac{1}{2}c^2 & \\textrm{otherwise.}
        \end{cases}

    Args:
        x (dynet.Expression): The first input expression
        y (dynet.Expression): The second input expression
    
    Keyword Arguments:
        c (number): The parameter of the huber distance parameterizing the cuttoff (default: (1.345))
    
    Returns:
        dynet.Expression: The huber distance
    """
    ensure_freshness(y);
    return Expression.from_cexpr(x.cg_version, c_huber_distance(x.c(), y.c(), c))
#expr-unsigned
cpdef Expression kmax_pooling(Expression x, unsigned k, unsigned d=1):
    """Kmax-pooling operation
    
    Select out k maximum values along a given dimension, in the same order as they appear. This will result in the size of the given dimension being changed to k.
    
    Args:
        x (dynet.Expression): 
        unsigned k (dynet.Expression): Number of maximum values to retrieve along the given dimension

    Keyword Arguments:
        unsigned d (int): Dimension on which to perform kmax-pooling (default: (1))
    
    Returns:
        dynet.Expression: 
    """
    return Expression.from_cexpr(x.cg_version, c_kmax_pooling(x.c(), k, d))
cpdef Expression pickneglogsoftmax(Expression x, unsigned v):
    """Negative softmax log likelihood
    
    This function takes in a vector of scores  :code:`x`, and performs a log softmax, takes the negative, and selects the likelihood corresponding to the element :code:`v`. This is perhaps the most standard loss function for training neural networks to predict one out of a set of elements.

    Args:
        x (dynet.Expression): Input scores
        v (int): True class
    
    Returns:
        dynet.Expression: :math:`-\log\left(\\frac{e^{x_v}}{\sum_j e^{x_j}}\\right)`
    """
    return Expression.from_cexpr(x.cg_version, c_pickneglogsoftmax(x.c(), v))
cpdef Expression pickneglogsoftmax_batch(Expression x, vector[unsigned] vs):
    """Negative softmax log likelihood on a batch
    
    This function takes in a batched vector of scores :code:`x`, and performs a log softmax, takes the negative, and selects the likelihood corresponding to the elements :code:`vs`. This is perhaps the most standard loss function for training neural networks to predict one out of a set of elements.
    
    Args:
        x (dynet.Expression): Input scores
        v (list): True classes
    
    Returns:
        dynet.Expression: :math:`-\sum_{v\in \\texttt{vs}}\log\left(\\frac{e^{x_v}}{\sum_j e^{x_j}}\\right)`
    """
    return Expression.from_cexpr(x.cg_version, c_pickneglogsoftmax(x.c(), vs))
cpdef Expression hinge(Expression x, unsigned v, float m=1.0):
    """Hinge loss
    
    This function takes in a vector of scores  :code:`x`, and calculates a hinge loss such that the element :code:`v` must be greater than all other elements by at least :code:`m`, otherwise a loss is incurred.

    Args:
        x (dynet.Expression): Input scores
        v (int): True class
        m (float): The margin
    
    Returns:
        dynet.Expression: :math:`\\sum_{\\tilde{v} != v} max(x_{\\tilde{v}} - x_v + m, 0)`
    """
    return Expression.from_cexpr(x.cg_version, c_hinge(x.c(), v, m))
cpdef Expression hinge_batch(Expression x, vector[unsigned] vs, float m=1.0):
    """Hinge loss on a batch
    
    This function takes in a batched vector of scores  :code:`xs`, and calculates a hinge loss such that the elements :code:`vs` must be greater than all other elements by at least :code:`m`, otherwise a loss is incurred.
    
    Args:
        x (dynet.Expression): Input scores
        v (list): True classes
        m (float): The margin
    
    Returns:
        dynet.Expression: The batched hinge loss function
    """
    return Expression.from_cexpr(x.cg_version, c_hinge(x.c(), vs, m))
cpdef Expression hinge_dim(Expression x, vector[unsigned] v, unsigned d=0, float m=1.0):
    """Dimensionwise hinge loss
    
    This function takes in a matrix of scores  :code:`x`, and calculates a hinge loss such that the elements :code:`v` must be greater than all other elements in dimension :code:`d` by at least :code:`m`, otherwise a loss is incurred.

    Args:
        x (dynet.Expression): Input scores
        v (list): True classes (size of the non-:code:`d` dimension)
        d (int): Dimension over which to perform the hinge loss
        m (float): The margin
    
    Returns:
        dynet.Expression: Containing a vector of losses the size of the non-:code:`d` dimension
    """
    return Expression.from_cexpr(x.cg_version, c_hinge_dim(x.c(), v, d, m))
# cpdef Expression hinge_dim_batch(Expression x, vector[vector[unsigned]] vs, unsigned d=0, float m=1.0):
#     """Dimensionwise hinge loss on a batch
#     
#     The batched version of :code:`hinge_dim`, where we pass in a list of lists of IDs :code:`v`, where each list corresponds to the IDs for one batch.
#     
#     Args:
#         x (dynet.Expression): Input scores
#         v (list[list]): True classes
#         d (int): Dimension over which to perform the hinge loss
#         m (float): The margin
#     
#     Returns:
#         dynet.Expression: The batched hinge_dim loss function
#     """
#     return Expression.from_cexpr(x.cg_version, c_hinge_dim(x.c(), vs, d, m))


cpdef Expression kmh_ngram(Expression x, unsigned v):
    """[summary]
    
    [description]
    
    Args:
        x (dynet.Expression): 
        v (dynet.Expression): 
    
    Returns:
        dynet.Expression: 
    """
    return Expression.from_cexpr(x.cg_version, c_kmh_ngram(x.c(), v))
cpdef Expression pick_range(Expression x, unsigned s, unsigned e, unsigned d = 0):
    """Pick range of elements
    
    Pick a range of elements from an expression.
    
    Args:
        x (dynet.Expression): input expression
        s (int): Start index
        e (int): End index
        d (int): Dimension along which to pick
    
    Returns:
        dynet.Expression: The value of {x[v],...,x[u]} along the dimension
    """
    return Expression.from_cexpr(x.cg_version, c_pick_range(x.c(), s, e, d))
# This is deprecated
cpdef Expression pickrange(Expression x, unsigned s, unsigned e):
    return Expression.from_cexpr(x.cg_version, c_pick_range(x.c(), s, e, 0))
cpdef Expression pick_batch_elem(Expression x, unsigned v):
    """Pick batch element.

    Pick batch element from a batched expression. For a Tensor with 3 batch elements:

    .. math::

        \\begin{pmatrix}
            x_{1,1,1} & x_{1,1,2} \\\\
            x_{1,2,1} & x_{1,2,2} \\\\
        \end{pmatrix}\\\\
        \\begin{pmatrix}
            x_{2,1,1} & x_{2,1,2} \\\\
            x_{2,2,1} & x_{2,2,2} \\\\
        \end{pmatrix}\\\\
        \\begin{pmatrix}
            x_{3,1,1} & x_{3,1,2} \\\\
            x_{3,2,1} & x_{3,2,2} \\\\
        \end{pmatrix}

    :code:`pick_batch_elem(t, 1)` will return a Tensor of

    .. math::

        \\begin{pmatrix}
            x_{2,1,1} & x_{2,1,2} \\\\
            x_{2,2,1} & x_{2,2,2} \\\\
        \end{pmatrix}
    
    Args:
        x (dynet.Expression): Input expression
        v (int): The index of the batch element to be picked.
    
    Returns:
        dynet.Expression: The expression of picked batch element. The picked element is a tensor whose batch dimension equals to one.
    """
    return Expression.from_cexpr(x.cg_version, c_pick_batch_elem(x.c(), v))
cpdef Expression pick_batch_elems(Expression x, vector[unsigned] vs):
    """Pick batch element.

    Pick batch element from a batched expression. For a Tensor with 3 batch elements:

    .. math::

        \\begin{pmatrix}
            x_{1,1,1} & x_{1,1,2} \\\\
            x_{1,2,1} & x_{1,2,2} \\\\
        \end{pmatrix}\\\\
        \\begin{pmatrix}
            x_{2,1,1} & x_{2,1,2} \\\\
            x_{2,2,1} & x_{2,2,2} \\\\
        \end{pmatrix}\\\\
        \\begin{pmatrix}
            x_{3,1,1} & x_{3,1,2} \\\\
            x_{3,2,1} & x_{3,2,2} \\\\
        \end{pmatrix}

    :code:`pick_batch_elems(t, [2, 3])` will return a Tensor of

    .. math::

        \\begin{pmatrix}
            x_{2,1,1} & x_{2,1,2} \\\\
            x_{2,2,1} & x_{2,2,2} \\\\
        \end{pmatrix}\\\\
        \\begin{pmatrix}
            x_{3,1,1} & x_{3,1,2} \\\\
            x_{3,2,1} & x_{3,2,2} \\\\
        \end{pmatrix}
    
    
    Args:
        x (dynet.Expression): Input expression
        vs (list): A list of indices of the batch elements to be picked.
    
    Returns:
        dynet.Expression: The expression of picked batch elements. The batch elements is a tensor whose batch dimension equals to the size of list `v`.
    """
    return Expression.from_cexpr(x.cg_version, c_pick_batch_elems(x.c(), vs))
#expr-float
cpdef Expression noise(Expression x, float stddev):
    """Additive gaussian noise
    
    Add gaussian noise to an expression.
    
    Args:
        x (dynet.Expression): Input expression
        stddev (number): The standard deviation of the gaussian
    
    Returns:
        dynet.Expression: :math:`y\sim\mathcal N(x,\\texttt{stddev})`
    """
    return Expression.from_cexpr(x.cg_version, c_noise(x.c(), stddev))

cpdef Expression dropout(Expression x, float p):
    """Dropout
    
    With a fixed probability, drop out (set to zero) nodes in the input expression, and **scale** the remaining nodes by 1/p. Note that there are `two kinds of dropout <http://cs231n.github.io/neural-networks-2/#reg>`_: 

    - *Regular dropout:* where we perform dropout at training time and then scale outputs by p at test time. 
    - *Inverted dropout:* where we perform dropout and scaling at training time, and do not need to do anything at test time. 

    DyNet implements the latter, so you only need to apply dropout at training time, and do not need to perform scaling and test time.
    
    Args:
        x (dynet.Expression): Input expression
        p (number): The dropout probability
    
    Returns:
        dynet.Expression: The dropped out expression :math:`y=\\frac{1}{1-\\texttt{p}}x\circ z, z\sim\\text{Bernoulli}(1-\\texttt{p})`
    """
    return Expression.from_cexpr(x.cg_version, c_dropout(x.c(), p))

    
cpdef Expression dropout_batch(Expression x, float p):
    """Dropout entire elements of a minibatch
    
    Identical to the dropout operation except entire batch elements are dropped

    Args:
        x (dynet.Expression): Input expression
        p (number): The dropout probability
    
    Returns:
        dynet.Expression: The dropped expression
    """
    return Expression.from_cexpr(x.cg_version, c_dropout_batch(x.c(), p))

cpdef Expression dropout_dim(Expression x, unsigned d, float p):
    """Dropout along one dimension
    
    Identical to the dropout operation except the dropout mask is the same across one dimension. Use this if you want to drop columns or lines in a matrix for example 

    For now this only supports tensors of order <= 3 (with or without batch dimension)

    Args:
        x (dynet.Expression): Input expression
        d (int): Dimension along which to drop
        p (number): The dropout probability
    
    Returns:
        dynet.Expression: The dropped expression
    """
    return Expression.from_cexpr(x.cg_version, c_dropout_dim(x.c(), d, p))

cpdef Expression block_dropout(Expression x, float p):
    """Block dropout
    
    Identical to the dropout operation, but either drops out *all* or *no* values in the expression, as opposed to making a decision about each value individually.
    
    Args:
        x (dynet.Expression): Input expression
        p (number): The dropout probability
    
    Returns:
        dynet.Expression: The block dropout expression
    """
    return Expression.from_cexpr(x.cg_version, c_block_dropout(x.c(), p))
#expr-dim
cpdef Expression reshape(Expression x, tuple d, unsigned int batch_size=1):
    """Reshape to another size
    
    This node reshapes a tensor to another size, without changing the underlying layout of the data. The layout of the data in DyNet is column-major, so if we have a 3x4 matrix :

    .. math::

        \\begin{pmatrix}
            x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\\\
            x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\\\
            x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\\\
        \end{pmatrix}

    and transform it into a 2x6 matrix, it will be rearranged as:

    .. math::

        \\begin{pmatrix}
            x_{1,1} & x_{3,1} & x_{2,2} & x_{1,3} & x_{3,3} & x_{2,4} \\\\
            x_{2,1} & x_{1,2} & x_{3,2} & x_{2,3} & x_{1,4} & x_{3,4} \\\\
        \end{pmatrix}

    **Note:** This is O(1) for forward, and O(n) for backward.
    
    Args:
        x (dynet.Expression): Input expression
        d (tuple): New dimension
    
    Keyword Arguments:
        batch_size (int): New batch size (default: (1))
    
    Returns:
        dynet.Expression: The reshaped expression
    """
    return Expression.from_cexpr(x.cg_version, c_reshape(x.c(),Dim(d, batch_size)))

cpdef Expression max_dim(Expression x, unsigned d=0):
    """Max out through a dimension
    
    Select out a element/row/column/sub-tensor from an expression, with maximum value along a given dimension. This will result in the dimension of the expression being reduced by 1.
    
    Args:
        x (dynet.Expression): Input expression
    
    Keyword Arguments:
        d (int): Dimension on which to perform the maxout (default: (0))
    
    Returns:
        dynet.Expression: An expression of sub-tensor with max value along dimension :code:`d`
    """
    return Expression.from_cexpr(x.cg_version, c_max_dim(x.c(), d))
cpdef Expression min_dim(Expression x, unsigned d=0):
    """Min out through a dimension
    
    Select out a element/row/column/sub-tensor from an expression, with minimum value along a given dimension. This will result in the dimension of the expression being reduced by 1.
    
    Args:
        x (dynet.Expression): Input expression
    
    Keyword Arguments:
        d (int): Dimension on which to perform the minout (default: (0))
    
    Returns:
        dynet.Expression: An expression of sub-tensor with min value along dimension :code:`d`
    """
    return Expression.from_cexpr(x.cg_version, c_min_dim(x.c(), d))

cpdef Expression contract3d_1d(Expression x, Expression y):
    """Contracts a rank 3 tensor and a rank 1 tensor into a rank 2 tensor
    
    The resulting tensor :math:`z` has coordinates :math:`z_ij = \sum_k x_{ijk} y_k`
    
    Args:
        x (dynet.Expression): Rank 3 tensor
        y (dynet.Expression): Vector
    
    Returns:
        Matrix
        dynet.Expression
    """
    ensure_freshness(y)
    return Expression.from_cexpr(x.cg_version, c_contract3d_1d(x.c(),y.c()))

cpdef Expression contract3d_1d_bias(Expression x, Expression y, Expression b):
    """Same as :code:`contract3d_1d` with an additional bias parameter

    The resulting tensor :math:`z` has coordinates :math:`z_{ij} = b_{ij}+\sum_k x_{ijk} y_k`

    Args:
        x (dynet.Expression): Rank 3 tensor
        y (dynet.Expression): Vector
        b (dynet.Expression): Bias vector
    
    Returns:
        Matrix
        dynet.Expression
    """
    ensure_freshness(y)
    ensure_freshness(b)
    return Expression.from_cexpr(x.cg_version, c_contract3d_1d(x.c(),y.c(),b.c()))

cpdef Expression contract3d_1d_1d(Expression x, Expression y, Expression z):
    """Contracts a rank 3 tensor and two rank 1 tensor into a rank 1 tensor

    This is the equivalent of calling :code:`contract3d_1d` and then performing a matrix vector multiplication.

    The resulting tensor :math:`t` has coordinates :math:`t_i = \sum_{j,k} x_{ijk} y_k z_j`
    
    Args:
        x (dynet.Expression): Rank 3 tensor
        y (dynet.Expression): Vector
        z (dynet.Expression): Vector
    
    Returns:
        Vector
        dynet.Expression
    """
    ensure_freshness(y)
    ensure_freshness(z)
    return Expression.from_cexpr(x.cg_version, c_contract3d_1d_1d(x.c(),y.c(),z.c()))

cpdef Expression contract3d_1d_1d_bias(Expression x, Expression y, Expression z, Expression b):
    """Same as :code:`contract3d_1d_1d` with an additional bias parameter
 
    This is the equivalent of calling :code:`contract3d_1d` and then performing an affine transform.

    The resulting tensor :math:`t` has coordinates :math:`t_i = b_i + \sum_{j,k} x_{ijk} y_k z_j`

    Args:
        x (dynet.Expression): Rank 3 tensor
        y (dynet.Expression): Vector
        z (dynet.Expression): Vector
        b (dynet.Expression): Bias vector
    
    Returns:
        Vector
        dynet.Expression
    """
    ensure_freshness(y)
    ensure_freshness(z)
    ensure_freshness(b)
    return Expression.from_cexpr(x.cg_version, c_contract3d_1d_1d(x.c(),y.c(),z.c(),b.c()))


cpdef Expression esum(list xs):
    """Sum
    
    This performs an elementwise sum over all the expressions in :code:`xs`
    
    Args:
        xs (list): A list of expression of same dimension
    
    Returns:
        dynet.Expression: An expression where the ith element is equal to :math:`\sum_{j=0}\\texttt{xs[}j\\texttt{][}i\\texttt{]}`
    """
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
    """Log, sum, exp
    
    The elementwise "logsumexp" function that calculates :math:`\ln(\sum_i e^{xs_i})`, used in adding probabilities in the log domain.
    
    Args:
        xs (list): A list of expression of same dimension

    Returns:
        dynet.Expression: An expression where the ith element is equal to :math:`\ln\left(\sum_{j=0}e^{\\texttt{xs[}j\\texttt{][}i\\texttt{]}}\\right)`
    """
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
    """Average
    
    This performs an elementwise average over all the expressions in :code:`xs`
    
    Args:
        xs (list): A list of expression of same dimension
    
    Returns:
        dynet.Expression: An expression where the ith element is equal to :math:`\\frac{1}{\\texttt{len(xs)}}\sum_{j=0}\\texttt{xs[}j\\texttt{][}i\\texttt{]}`
    """
    assert xs, 'List is empty, nothing to average.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs: 
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_average(cvec))

cpdef Expression emax(list xs):
    """Max
    
    This performs an elementwise max over all the expressions in :code:`xs`

    Args:
        xs (list): A list of expression of same dimension
    
    Returns:
        dynet.Expression: An expression where the ith element is equal to :math:`\max_j\\texttt{xs[}j\\texttt{][}i\\texttt{]}`
    """
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
    """Concatenate columns
    
    Perform a concatenation of the columns in multiple expressions. All expressions must have the same number of rows.
    
    Args:
        xs (list): A list of expressions
    
    Returns:
        dynet.Expression: The expression with the columns concatenated
    """
    assert xs, 'List is empty, nothing to concatenate.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs:
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_concat_cols(cvec))

cpdef Expression concatenate(list xs, unsigned d=0):
    """Concatenate
    
     Perform a concatenation of multiple expressions along a particular dimension.
     All expressions must have the same dimensions except for the dimension to be concatenated (rows by default).
    
    Args:
        xs (list): A list of expressions
        d: The dimension along with to perform concatenation
    
    Returns:
        dynet.Expression: The expression concatenated along the particular dimension
    """
    assert xs, 'List is empty, nothing to concatenate.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs:
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_concat(cvec, d))

cpdef Expression concatenate_to_batch(list xs):
    """Concatenate list of expressions to a single batched expression
    
    Perform a concatenation of several expressions along the batch dimension. All expressions must have the same shape except for the batch dimension.

    Args:
        xs (list): A list of expressions of same dimension (except batch size)
    
    Returns:
        dynet.Expression: The expression with the batch dimensions concatenated
    """
    assert xs, 'List is empty, nothing to concatenate.'
    cdef vector[CExpression] cvec
    cdef Expression x
    for x in xs:
        ensure_freshness(x) 
        cvec.push_back(x.c())
    return Expression.from_cexpr(x.cg_version, c_concat_to_batch(cvec))

cpdef Expression affine_transform(list exprs):
    """Affine transform
    
    This performs an affine transform over an arbitrary (odd) number of expressions held in the input initializer list xs. The first expression is the "bias," which is added to the expression as-is. The remaining expressions are multiplied together in pairs, then added. A very common usage case is the calculation of the score for a neural network layer (e.g. :math:`b + Wz`) where b is the bias, W is the weight matrix, and z is the input. In this case :code:`xs[0] = b`, :code:`xs[1] = W`, and :code:`xs[2] = z`.
     
    Args:
        exprs (list): A list containing an odd number of expressions
    
    Returns:
        dynet.Expression: An expression equal to: :code:`xs[0] + xs[1]*xs[2] + xs[3]*xs[4] + ...`
    """
    assert exprs, 'List input to affine_transform must not be empty.'
    cdef Expression e
    cdef vector[CExpression] ves
    for e in exprs:
        ensure_freshness(e) 
        ves.push_back(e.c())
    return Expression.from_cexpr(e.cg_version, c_affine_transform(ves))

cpdef Expression layer_norm(Expression x, Expression g, Expression b):
    """Layer normalization

    Performs layer normalization : 

    .. math::

        \\begin{split}
           \mu &= \\frac 1 n \sum_{i=1}^n x_i\\\\
           \sigma &= \sqrt{\\frac 1 n \sum_{i=1}^n (x_i-\mu)^2}\\\\
           y&=\\frac {\\boldsymbol{g}} \sigma \circ (\\boldsymbol{x}-\mu) + \\boldsymbol{b}\\\\
        \end{split}
 
        
    Reference : `Ba et al., 2016 <http://arxiv.org/abs/1607.06450>`_
        
    Args:
        x (dynet.Expression): Input expression (possibly batched)
        g (dynet.Expression): Gain (same dimension as x, no batch dimension)
        b (dynet.Expression): Bias (same dimension as x, no batch dimension)
    
    Returns:
        An expression of the same dimension as :code:`x`
        dynet.Expression
    """
    ensure_freshness(g)
    ensure_freshness(b)
    return Expression.from_cexpr(x.cg_version, c_layer_norm(x.c(),g.c(),b.c()))

cpdef Expression weight_norm(Expression w, Expression g):
    """Weight normalization

    Performs weight normalization : 

    .. math::

        \\begin{split}
           \hat{w} &= g\\frac{w}{\Vert w\Vert}\\\\
        \end{split}
 
        
    Reference : `Salimans, Kingma 2016 <https://arxiv.org/abs/1602.07868>`_
        
    Args:
        w (dynet.Expression): Input expression (weight parameter)
        g (dynet.Expression): Gain (scalar expression, usually also a parameter)
    
    Returns:
        An expression of the same dimension as :code:`w`
        dynet.Expression
    """
    ensure_freshness(g)
    return Expression.from_cexpr(w.cg_version, c_weight_norm(w.c(),g.c()))

cpdef Expression vanilla_lstm_gates(Expression x_t, Expression h_tm1, Expression Wx, Expression Wh, Expression b, weightnoise_std=0.0):
    """Computes LSTM gates (matrix multiply + nonlinearities):
    
       gates_i = sigmoid (Wx_i * x_t + Wh_i * h_tm1 + b_i)
       gates_f = sigmoid (Wx_f * x_t + Wh_f * h_tm1 + b_f + 1)
       gates_o = sigmoid (Wx_o * x_t + Wh_o * h_tm1 + b_o)
       gates_g =   tanh  (Wx_g * x_t + Wh_g * h_tm1 + b_g)
       
       Where optionally gaussian noise with the given standard deviation is applied to Wx, Wh, b parameters. 
       
       returns [gates_i]
               [gates_f]
               [gates_o]
               [gates_g]

    Args:
        x_t (dynet.Expression): Input at current timestep (vector size I)
        h_tm1 (dynet.Expression): State previous timestep (vector size H)
        Wx (dynet.Expression): Parameter matrix size 4H x I
        Wh (dynet.Expression): Parameter matrix size 4H x H
        b (dynet.Expression): Bias parameter size 4H
        weightnoise_std (real): apply gaussian noise to weights (Wx, Wh, b); requires only temporary additional memory
    
    Returns:
        Vector size 4H
        dynet.Expression
    """
    ensure_freshness(x_t)
    ensure_freshness(h_tm1)
    return Expression.from_cexpr(x_t.cg_version, c_vanilla_lstm_gates(x_t.c(),h_tm1.c(),Wx.c(),Wh.c(),b.c(), weightnoise_std))

cpdef Expression vanilla_lstm_c(Expression c_tm1, Expression gates_t):
    """Computes LSTM cell: c_t = gates_i . gates_g + gates_f . c_tm1

    Args:
        c_tm1 (dynet.Expression): Cell at previous timestep (vector size H)
        gates_t (dynet.Expression): Gates at current timestep as computed by vanilla_lstm_gates (vector size 4H)
    
    Returns:
        Vector size H
        dynet.Expression
    """
    ensure_freshness(c_tm1)
    ensure_freshness(gates_t)
    return Expression.from_cexpr(c_tm1.cg_version, c_vanilla_lstm_c(c_tm1.c(),gates_t.c()))

cpdef Expression vanilla_lstm_h(Expression c_t, Expression gates_t):
    """Computes LSTM output: h_t = o_t . tanh(c_t)

    Args:
        c_t (dynet.Expression): Cell at current timestep (vector size H)
        gates_t (dynet.Expression): Gates at current timestep as computed by vanilla_lstm_gates (vector size 4H)
    
    Returns:
        Vector size H
        dynet.Expression
    """
    ensure_freshness(c_t)
    ensure_freshness(gates_t)
    return Expression.from_cexpr(c_t.cg_version, c_vanilla_lstm_h(c_t.c(),gates_t.c()))

# }}}
    
# {{{ RNNS / Builders
# TODO: unify these with inheritance

cdef class _RNNBuilder: # {{{
    """
    """
    cdef CRNNBuilder *thisptr
    cdef RNNState _init_state
    cdef int cg_version 

    def __dealloc__(self):
        del self.thisptr

    cpdef set_dropout(self, float f):
        """[summary]
        
        [description]
        
        Args:
            float f: [description]
        """
        self.thisptr.set_dropout(f)
    cpdef disable_dropout(self):
        """[summary]
        
        [description]
        """
        self.thisptr.disable_dropout()

    cdef new_graph(self, update=True):
        self.thisptr.new_graph(_cg.thisptr[0], update)
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

    cpdef RNNState initial_state(self,vecs=None,update=True):
        """Get a :code:`dynet.RNNState`
        
        This initializes a :code:`dynet.RNNState` by loading the parameters in the computation graph
        
        Args:
            vecs (list): Initial hidden state for each layer as a list of :code:`dynet.Expression` s  (default: {None})
            update (bool): trainer updates internal parameters (default: {True})
        
        Returns:
            :code:`dynet.RNNState` used to feed inputs/transduces sequences, etc...
            dynet.RNNState
        """
        if self.cg_version != _cg.version():
            self.new_graph(update)
            if vecs is not None:
                self.start_new_sequence(vecs)
            else:
                self.start_new_sequence()
            self._init_state = RNNState(self, -1)
        return self._init_state

    cpdef RNNState initial_state_from_raw_vectors(self,vecs=None, update=True):
        """Get a :code:`dynet.RNNState`
        
        This initializes a :code:`dynet.RNNState` by loading the parameters in the computation graph

        Use this if you want to initialize the hidden states with values directly rather than expressions.
        
        Args:
            vecs (list): Initial hidden state for each layer as a list of numpy arrays  (default: {None})
            update (bool): trainer updates internal parameters (default: {True})
        
        Returns:
            :code:`dynet.RNNState` used to feed inputs/transduces sequences, etc...
            dynet.RNNState
        """
        if self.cg_version != _cg.version():
            self.new_graph(update)
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

    cpdef ParameterCollection param_collection(self):
        return ParameterCollection.wrap(self.thisptr.get_parameter_collection())
# _RNNBuilder }}}

cdef class SimpleRNNBuilder(_RNNBuilder): # {{{
    """[summary]
    
    [description]
    """
    cdef CSimpleRNNBuilder* thissimpleptr
    cdef tuple _spec
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, ParameterCollection model):
        self._spec = (layers, input_dim, hidden_dim)
        if layers > 0:
            self.thissimpleptr = self.thisptr = new CSimpleRNNBuilder(layers, input_dim, hidden_dim, model.thisptr)
        else:
            self.thissimpleptr = self.thisptr = new CSimpleRNNBuilder()
        self.cg_version = -1

    @property
    def spec(self): return self._spec
    @classmethod
    def from_spec(cls, spec, model):
        return SimpleRNNBuilder(*spec, model)

# TODO rename to parameters()?
    cpdef get_parameters(self):
        """Retrieve the internal parameters of the RNN
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{hx},W_{hh},b_h`
        
        Returns:
            List of parameters for each layer
            list
        """
        params = []
        for l in self.thissimpleptr.params:
            layer_params=[]
            for w in l:
                layer_params.append(Parameters.wrap_ptr(w))
            params.append(layer_params)
        return params

# TODO rename to parameter_expressions()?
    cpdef get_parameter_expressions(self):
        """Retrieve the internal parameters expressions of the RNN
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{hx},W_{hh},b_h`
        
        Returns:
            List of parameter expressions for each layer
            list

        Raises:
            ValueError: This raises an expression if initial_state hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to :code:`dynet.parameter(rnn.get_parameters()[0][0])` for example).
        """
        if self.thissimpleptr.param_vars.size() == 0 or self.thissimpleptr.param_vars[0][0].is_stale():
            raise ValueError("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression")

        exprs = []
        for l in self.thissimpleptr.param_vars:
            layer_exprs=[]
            for w in l:
                layer_exprs.append(Expression.from_cexpr(_cg.version(),w))
            exprs.append(layer_exprs)
        return exprs

    def whoami(self): return "SimpleRNNBuilder"
# SimpleRNNBuilder }}}
    
cdef class GRUBuilder(_RNNBuilder): # {{{
    """[summary]
    
    [description]
    """
    cdef CGRUBuilder* thisgruptr
    cdef tuple _spec
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, ParameterCollection model):
        _spec = (layers, input_dim, hidden_dim)
        if layers > 0:
            self.thisgruptr = self.thisptr = new CGRUBuilder(layers, input_dim, hidden_dim, model.thisptr)
        else:
            self.thisgruptr = self.thisptr = new CGRUBuilder()
        self.cg_version = -1

    @property
    def spec(self): return self._spec
    @classmethod
    def from_spec(cls, spec, model):
        return GRUBuilder(*spec, model)

# TODO rename to parameters()?
    cpdef get_parameters(self):
        """Retrieve the internal parameters of the GRU
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{zx},W_{zh},b_z,W_{rx},W_{rh},b_r,W_{hx},W_{hh},b_h`
        
        Returns:
            List of parameters for each layer
            list
        """
        params = []
        for l in self.thisgruptr.params:
            layer_params=[]
            for w in l:
                layer_params.append(Parameters.wrap_ptr(w))
            params.append(layer_params)
        return params

# TODO rename to parameter_expressions()?
    cpdef get_parameter_expressions(self):
        """Retrieve the internal parameters expressions of the GRU
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{zx},W_{zh},b_z,W_{rx},W_{rh},b_r,W_{hx},W_{hh},b_h`
        
        Returns:
            List of parameter expressions for each layer
            list

        Raises:
            ValueError: This raises an expression if initial_state hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to :code:`dynet.parameter(rnn.get_parameters()[0][0])` for example).
        """
        if self.thisgruptr.param_vars.size() == 0 or self.thisgruptr.param_vars[0][0].is_stale():
            raise ValueError("Attempt to use a stale expression, renew CG and/or call initial_state before accessing GRUBuilder internal parameters expression")

        exprs = []
        for l in self.thisgruptr.param_vars:
            layer_exprs=[]
            for w in l:
                layer_exprs.append(Expression.from_cexpr(_cg.version(),w))
            exprs.append(layer_exprs)
        return exprs

    def whoami(self): return "GRUBuilder"
# GRUBuilder }}}

cdef class LSTMBuilder(_RNNBuilder): # {{{
    """[summary]
    
    [description]
    """
    cdef CLSTMBuilder* thislstmptr
    cdef tuple _spec
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, ParameterCollection model):
        self._spec = (layers, input_dim, hidden_dim)
        if layers > 0:
            self.thislstmptr = self.thisptr = new CLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr)
        else:
            self.thislstmptr = self.thisptr = new CLSTMBuilder()
        self.cg_version = -1

    @property
    def spec(self): return self._spec
    @classmethod
    def from_spec(cls, spec, model):
        return LSTMBuilder(*spec, model)

# TODO rename to parameters()?
    cpdef get_parameters(self):
        """Retrieve the internal parameters of the LSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`
        
        Returns:
            List of parameters for each layer
            list
        """
        params = []
        for l in self.thislstmptr.params:
            layer_params=[]
            for w in l:
                layer_params.append(Parameters.wrap_ptr(w))
            params.append(layer_params)
        return params


# TODO rename to parameter_expressions()?
    cpdef get_parameter_expressions(self):
        """Retrieve the internal parameters expressions of the LSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`
        
        Returns:
            List of parameter expressions for each layer
            list

        Raises:
            ValueError: This raises an expression if initial_state hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to :code:`dynet.parameter(rnn.get_parameters()[0][0])` for example).
        """
        if self.thislstmptr.param_vars.size() == 0 or self.thislstmptr.param_vars[0][0].is_stale():
            raise ValueError("Attempt to use a stale expression, renew CG and/or call initial_state before accessing LSTMBuilder internal parameters expression")

        exprs = []
        for l in self.thislstmptr.param_vars:
            layer_exprs=[]
            for w in l:
                layer_exprs.append(Expression.from_cexpr(_cg.version(),w))
            exprs.append(layer_exprs)
        return exprs

    def whoami(self): return "LSTMBuilder"
# LSTMBuilder }}}

cdef class VanillaLSTMBuilder(_RNNBuilder): # {{{
    """VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections
    
    This cell runs according to the following dynamics :

    .. math::

        \\begin{split}
            i_t & =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+b_i)\\\\
            f_t & = \sigma(W_{fx}x_t+W_{fh}h_{t-1}+b_f+1)\\\\
            o_t & = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+b_o)\\\\
            \\tilde{c_t} & = \\tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)\\\\
            c_t & = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t\\\\
            h_t & = \\tanh(c_t)\circ o_t\\\\
        \end{split}

    Args:
        layers (int): Number of layers
        input_dim (int): Dimension of the input
        hidden_dim (int): Dimension of the recurrent units
        model (dynet.ParameterCollection): ParameterCollection to hold the parameters
        ln_lstm (bool): Whether to use layer normalization

    """
    cdef CVanillaLSTMBuilder* thisvanillaptr
    cdef tuple _spec
    def __init__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, ParameterCollection model, ln_lstm=False):
        self._spec = (layers, input_dim, hidden_dim, ln_lstm)
        if layers > 0:
            self.thisvanillaptr = self.thisptr = new CVanillaLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr, ln_lstm)
        else:
            self.thisvanillaptr = self.thisptr = new CVanillaLSTMBuilder()
        self.cg_version = -1

    @property
    def spec(self): return self._spec

    @classmethod
    def from_spec(cls, spec, model):
        layers, input_dim, hidden_dim, ln_lstm = spec
        return VanillaLSTMBuilder(layers, input_dim, hidden_dim, model, ln_lstm)

# TODO rename to parameters()?
    cpdef get_parameters(self):
        """Retrieve the internal parameters of the VanillaLSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_x,W_h,b` where :math:`W_x,W_h` are stacked version of the individual gates matrices:

        .. code-block:: text

                  h/x   
                +------+
                |      |
            i   |      |
                +------+
                |      |
            f   |      |
                +------+
                |      |
            o   |      |
                +------+
                |      |
            c   |      |
                +------+

        Returns:
            List of parameters for each layer
            list
        """
        params = []
        for l in self.thisvanillaptr.params:
            layer_params=[]
            for w in l:
                layer_params.append(Parameters.wrap_ptr(w))
            params.append(layer_params)
        return params

# TODO rename to parameter_expressions()?
    cpdef get_parameter_expressions(self):
        """Retrieve the internal parameters expressions of the VanillaLSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_x,W_h,b` where :math:`W_x,W_h` are stacked version of the individual gates matrices:

        .. code-block:: text

                  h/x   
                +------+
                |      |
            i   |      |
                +------+
                |      |
            f   |      |
                +------+
                |      |
            o   |      |
                +------+
                |      |
            c   |      |
                +------+
        
        Returns:
            List of parameter expressions for each layer
            list

        Raises:
            ValueError: This raises an expression if initial_state hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to :code:`dynet.parameter(rnn.get_parameters()[0][0])` for example).
        """
        if self.thisvanillaptr.param_vars.size() == 0 or self.thisvanillaptr.param_vars[0][0].is_stale():
            raise ValueError("Attempt to use a stale expression, renew CG and/or call initial_state before accessing VanillaLSTMBuilder internal parameters expression")

        exprs = []
        for l in self.thisvanillaptr.param_vars:
            layer_exprs=[]
            for w in l:
                layer_exprs.append(Expression.from_cexpr(_cg.version(),w))
            exprs.append(layer_exprs)
        return exprs


    cpdef void set_dropouts(self, float d, float d_r):
        """Set the dropout rates
        
        The dropout implemented here is the variational dropout with tied weights introduced in `Gal, 2016 <http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks>`_

        More specifically, dropout masks :math:`\mathbf{z_x}\sim \\text(1-d_x)`, :math:`\mathbf{z_h}\sim \\text{Bernoulli}(1-d_h)` are sampled at the start of each sequence.

        The dynamics of the cell are then modified to :

        .. math::

            \\begin{split}
                i_t & =\sigma(W_{ix}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ih}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_i)\\\\
                f_t & = \sigma(W_{fx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{fh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_f)\\\\
                o_t & = \sigma(W_{ox}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{oh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_o)\\\\
                \\tilde{c_t} & = \tanh(W_{cx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ch}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_c)\\\\
                c_t & = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t\\\\
                h_t & = \\tanh(c_t)\circ o_t\\\\
            \end{split}

        For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation

        Args:
            d (number): Dropout rate :math:`d_x` for the input :math:`x_t`
            d_r (number): Dropout rate :math:`d_x` for the output :math:`h_t`
        """
        self.thisvanillaptr.set_dropout(d, d_r)

    cpdef void set_dropout_masks(self, unsigned batch_size=1):
        """Set dropout masks at the beginning of a sequence for a specific batch size
        
        If this function is not called on batched input, the same mask will be applied across all batch elements. Use this to apply different masks to each batch element

        You need to call this __AFTER__ calling `initial_state`
        
        Args:
            batch_size (int): Batch size (default: {1})
        """
        self.thisvanillaptr.set_dropout_masks(batch_size)

    def whoami(self): return "VanillaLSTMBuilder"
# VanillaLSTMBuilder }}}



cdef class CompactVanillaLSTMBuilder(_RNNBuilder): # {{{
    """CompactVanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections
    
    This cell runs according to the following dynamics :

    .. math::

        \\begin{split}
            i_t & =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+b_i)\\\\
            f_t & = \sigma(W_{fx}x_t+W_{fh}h_{t-1}+b_f+1)\\\\
            o_t & = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+b_o)\\\\
            \\tilde{c_t} & = \\tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)\\\\
            c_t & = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t\\\\
            h_t & = \\tanh(c_t)\circ o_t\\\\
        \end{split}

    Args:
        layers (int): Number of layers
        input_dim (int): Dimension of the input
        hidden_dim (int): Dimension of the recurrent units
        model (dynet.ParameterCollection): ParameterCollection to hold the parameters

    """
    cdef CCompactVanillaLSTMBuilder* thiscompvanillaptr
    cdef tuple _spec
    def __init__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, ParameterCollection model):
        self._spec = (layers, input_dim, hidden_dim)
        if layers > 0:
            self.thiscompvanillaptr = self.thisptr = new CCompactVanillaLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr)
        else:
            self.thiscompvanillaptr = self.thisptr = new CCompactVanillaLSTMBuilder()
        self.cg_version = -1

    @property
    def spec(self): return self._spec

    @classmethod
    def from_spec(cls, spec, model):
        layers, input_dim, hidden_dim = spec
        return CompactVanillaLSTMBuilder(layers, input_dim, hidden_dim, model)

# TODO rename to parameters()?
    cpdef get_parameters(self):
        """Retrieve the internal parameters of the CompactVanillaLSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_x,W_h,b` where :math:`W_x,W_h` are stacked version of the individual gates matrices:

        .. code-block:: text

                  h/x   
                +------+
                |      |
            i   |      |
                +------+
                |      |
            f   |      |
                +------+
                |      |
            o   |      |
                +------+
                |      |
            c   |      |
                +------+

        Returns:
            List of parameters for each layer
            list
        """
        params = []
        for l in self.thiscompvanillaptr.params:
            layer_params=[]
            for w in l:
                layer_params.append(Parameters.wrap_ptr(w))
            params.append(layer_params)
        return params

# TODO rename to parameter_expressions()?
    cpdef get_parameter_expressions(self):
        """Retrieve the internal parameters expressions of the CompactVanillaLSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_x,W_h,b` where :math:`W_x,W_h` are stacked version of the individual gates matrices:

        .. code-block:: text

                  h/x   
                +------+
                |      |
            i   |      |
                +------+
                |      |
            f   |      |
                +------+
                |      |
            o   |      |
                +------+
                |      |
            c   |      |
                +------+
        
        Returns:
            List of parameter expressions for each layer
            list

        Raises:
            ValueError: This raises an expression if initial_state hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to :code:`dynet.parameter(rnn.get_parameters()[0][0])` for example).
        """
        if self.thiscompvanillaptr.param_vars.size() == 0 or self.thiscompvanillaptr.param_vars[0][0].is_stale():
            raise ValueError("Attempt to use a stale expression, renew CG and/or call initial_state before accessing CompactVanillaLSTMBuilder internal parameters expression")

        exprs = []
        for l in self.thiscompvanillaptr.param_vars:
            layer_exprs=[]
            for w in l:
                layer_exprs.append(Expression.from_cexpr(_cg.version(),w))
            exprs.append(layer_exprs)
        return exprs


    cpdef void set_dropouts(self, float d, float d_r):
        """Set the dropout rates
        
        The dropout implemented here is the variational dropout with tied weights introduced in `Gal, 2016 <http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks>`_

        More specifically, dropout masks :math:`\mathbf{z_x}\sim \\text(1-d_x)`, :math:`\mathbf{z_h}\sim \\text{Bernoulli}(1-d_h)` are sampled at the start of each sequence.

        The dynamics of the cell are then modified to :

        .. math::

            \\begin{split}
                i_t & =\sigma(W_{ix}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ih}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_i)\\\\
                f_t & = \sigma(W_{fx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{fh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_f)\\\\
                o_t & = \sigma(W_{ox}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{oh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_o)\\\\
                \\tilde{c_t} & = \tanh(W_{cx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ch}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_c)\\\\
                c_t & = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t\\\\
                h_t & = \\tanh(c_t)\circ o_t\\\\
            \end{split}

        For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation

        Args:
            d (number): Dropout rate :math:`d_x` for the input :math:`x_t`
            d_r (number): Dropout rate :math:`d_x` for the output :math:`h_t`
        """
        self.thiscompvanillaptr.set_dropout(d, d_r)

    cpdef void set_dropout_masks(self, unsigned batch_size=1):
        """Set dropout masks at the beginning of a sequence for a specific batch size
        
        If this function is not called on batched input, the same mask will be applied across all batch elements. Use this to apply different masks to each batch element

        You need to call this __AFTER__ calling `initial_state`
        
        Args:
            batch_size (int): Batch size (default: {1})
        """
        self.thiscompvanillaptr.set_dropout_masks(batch_size)

    cpdef void set_weightnoise(self, float std):
        """Set the gaussian weight noise

        Args:
            std (number): Standard deviation of weight noise
        """
        self.thiscompvanillaptr.set_weightnoise(std)

    def whoami(self): return "CompactVanillaLSTMBuilder"
# CompactVanillaLSTMBuilder }}}




cdef class FastLSTMBuilder(_RNNBuilder): # {{{
    """[summary]
    
    [description]
    """
    cdef CFastLSTMBuilder* thisfastptr
    def __cinit__(self, unsigned layers, unsigned input_dim, unsigned hidden_dim, ParameterCollection model):
        self.thisfastptr = self.thisptr = new CFastLSTMBuilder(layers, input_dim, hidden_dim, model.thisptr)
        self.cg_version = -1

# TODO rename to parameters()?
    cpdef get_parameters(self):
        """Retrieve the internal parameters of the FastLSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`
        
        Returns:
            List of parameters for each layer
            list
        """
        params = []
        for l in self.thisfastptr.params:
            layer_params=[]
            for w in l:
                layer_params.append(Parameters.wrap_ptr(w))
            params.append(layer_params)
        return params

# TODO rename to parameter_expressions()?
    cpdef get_parameter_expressions(self):
        """Retrieve the internal parameters expressions of the FastLSTM
        
        The output is a list with one item per layer. Each item is a list containing :math:`W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`
       
        Returns:
            List of parameter expressions for each layer
            list

        Raises:
            ValueError: This raises an expression if initial_state hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to :code:`dynet.parameter(rnn.get_parameters()[0][0])` for example).
        """
        if self.thisfastptr.param_vars.size() == 0 or self.thisfastptr.param_vars[0][0].is_stale():
            raise ValueError("Attempt to use a stale expression, renew CG and/or call initial_state before accessing FastLSTMBuilder internal parameters expression")

        exprs = []
        for l in self.thisfastptr.param_vars:
            layer_exprs=[]
            for w in l:
                layer_exprs.append(Expression.from_cexpr(_cg.version(),w))
            exprs.append(layer_exprs)
        return exprs

    def whoami(self): return "FastLSTMBuilder"
# }}}

class BiRNNBuilder(object): # {{{
    """
    Builder for BiRNNs that delegates to regular RNNs and wires them together.  
    
        builder = BiRNNBuilder(1, 128, 100, model, LSTMBuilder)
        [o1,o2,o3] = builder.transduce([i1,i2,i3])
    """
    def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory, builder_layers=None):
        """Args:
            num_layers: depth of the BiRNN
            input_dim: size of the inputs
            hidden_dim: size of the outputs (and intermediate layer representations.) This hidden dim is split evenly between the two constituent RNNs, and thus must be even. 
            model
            rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
            builder_layers: list of (forward, backward) pairs of RNNBuilder instances to directly initialize layers
        """
        self.spec = num_layers, input_dim, hidden_dim, rnn_builder_factory, builder_layers
        model = self.model = model.add_subcollection("birnn")
        if builder_layers is None:
            assert num_layers > 0
            assert hidden_dim % 2 == 0, "BiRNN hidden dimension must be even."
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

    @classmethod
    def from_spec(cls, spec, model):
        num_layers, input_dim, hidden_dim, rnn_builder_factory, builder_layers = spec
        return cls(num_layers, input_dim, hidden_dim, model, rnn_builder_factory, builder_layers)

    def param_collection(self): return self.model

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
        Does not preserve the internal state after adding the inputs.
        Args:
            es (list): a list of Expression

        see also transduce(xs)

        code:`.transduce(xs)` is different from .add_inputs(xs) in the following way:

        - code:`.add_inputs(xs)` returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

        - :code:`.transduce(xs)` returns a list of Expression. These are just the output
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
# BiRNNBuilder }}}

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
        """Manually set the output :math:`h_t`
        
        Args:
            es (list): List of expressions, one for each layer (default: {None})
        
        Returns:
            New RNNState
            dynet.RNNState
        """
        cdef Expression res = self.builder.set_h(CRNNPointer(self.state_idx), es)
        cdef int state_idx = <int>self.builder.thisptr.state()
        return RNNState(self.builder, state_idx, self, res)

    cpdef RNNState set_s(self, es=None):
        """Manually set the hidden states
        
        This is different from :code:`set_h` because, for LSTMs for instance this also sets the cell state. The format is :code:`[new_c[0],...,new_c[n],new_h[0],...,new_h[n]]`
        
        Args:
            es (list): List of expressions, in this format : :code:`[new_c[0],...,new_c[n],new_h[0],...,new_h[n]]` (default: {None})
        
        Returns:
            New RNNState
            dynet.RNNState
        """
        cdef Expression res = self.builder.set_s(CRNNPointer(self.state_idx), es)
        cdef int state_idx = <int>self.builder.thisptr.state()
        return RNNState(self.builder, state_idx, self, res)

    cpdef RNNState add_input(self, Expression x):
        """This computes :math:`h_t = \\text{RNN}(x_t)`
        
        Args:
            x (dynet.Expression): Input expression
        
        Returns:
            New RNNState
            dynet.RNNState
        """
        cdef Expression res = self.builder.add_input_to_prev(CRNNPointer(self.state_idx), x)
        cdef int state_idx = <int>self.builder.thisptr.state()
        return RNNState(self.builder, state_idx, self, res)

    def add_inputs(self, xs):
        """Returns the list of states obtained by adding the given inputs to the current state, one by one.

        see also :code:`transduce(xs)`

        :code:`.transduce(xs)` is different from :code:`.add_inputs(xs)` in the following way:

        - :code:`.add_inputs(xs)` returns a list of RNNState. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (:code:`h()` and :code:`s()` )

        - :code:`.transduce(xs)` returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
        
        :code:`transduce` is much more memory efficient than :code:`add_inputs`.

        Args:
            xs (list): list of input expressions

        Returns:
            New RNNState
            dynet.RNNState
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
        
        see also :code:`add_inputs(xs)`

        :code:`.transduce(xs)` is different from :code:`.add_inputs(xs)` in the following way:

        - :code:`.add_inputs(xs)` returns a list of RNNState. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (:code:`h()` and :code:`s()` )

        - :code:`.transduce(xs)` returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
        
        :code:`transduce` is much more memory efficient than :code:`add_inputs`.

        Args:
            xs (list): list of input expressions

        Returns:
            New RNNState
            dynet.RNNState
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
        For LSTM, s() is a series of of memory vectors, followed the series followed by the series returned by h():

        .. code:: none

            (c[1],...,c[num_layers], h[1],...,h[num_layers])
        """
        return tuple(self.builder.get_s(CRNNPointer(self.state_idx)))

    cpdef RNNState prev(self):
        """Gets previous RNNState

        In case you need to rewind
        """
        return self._prev

    def b(self):
        """Get the underlying RNNBuilder
        
        In case you need to set dropout or other stuff.
        
        Returns:
            Underlying RNNBuilder
            dynet.RNNBuilder
        """
        return self.builder
# RNNState }}}

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

# RNNS / Builders }}}

# {{{ Trainers
cdef class Trainer:
    """
    Generic trainer

    Attributes:
        learning_rate(number): Global learning rate for all parameters 
    """
    cdef CTrainer *thisptr
    def __dealloc__(self):
        del self.thisptr
    cpdef update(self):
        """Update the parameters
        
        The update equation is different for each trainer, check the online c++ documentation for more details on what each trainer does
        """
        self.thisptr.update()

    cpdef update_subset(self, updated_params, updated_lookups):
        """Update a subset of parameters
        
        Only use this in last resort, a more elegant way to update only a subset of parameters is to use the "update" keyword in dy.parameter or Parameter.expr() to specify which parameters need to be updated __during the creation of the computation graph__
        
        Args:
            updated_params(list): Indices of parameters to update
            updated_lookups(list): Indices of lookup parameters to update
        """
        cdef vector[unsigned] uparamvec
        for i in updated_params: uparamvec.push_back(i)
        cdef vector[unsigned] ulookupvec
        for i in updated_lookups: ulookupvec.push_back(i)
        # self.thisptr.update(uparamvec, ulookupvec)
    cpdef update_epoch(self, r):
        """DEPRECATED: do not use.
        """
        self.thisptr.update_epoch(r)
    cpdef status(self):
        """Outputs information about the trainer in the stderr 
        
        (number of updates since last call, number of clipped gradients, learning rate, etc...)
        """
        self.thisptr.status()
    cpdef set_sparse_updates(self,bool su):
        """Sets updates to sparse updates

        DyNet trainers support two types of updates for lookup parameters, sparse and dense. Sparse updates are the default. They have the potential to be faster, as they only touch the parameters that have non-zero gradients. However, they may not always be faster (particulary on GPU with mini-batch training), and are not precisely numerically correct for some update rules such as MomentumTrainer and AdamTrainer. Thus, if you set this variable to false, the trainer will perform dense updates and be precisely correct, and maybe faster sometimes.
        Args:
            su(bool): flag to activate/deactivate sparse updates
        """
        self.thisptr.sparse_updates_enabled = su
    cpdef set_clip_threshold(self,float thr):
        """Set clipping thershold
        
        To deactivate clipping, set the threshold to be <=0
        
        Args:
            thr(number): Clipping threshold
        """
        if thr<=0:
            self.thisptr.clipping_enabled = False
            self.thisptr.clip_threshold = 0.0
        else:
            self.thisptr.clipping_enabled = True
            self.thisptr.clip_threshold = thr
    cpdef get_clip_threshold(self):
        """Get clipping threshold
        
        Returns:
            number: Gradient clipping threshold
        """
        return self.thisptr.clip_threshold

    @property
    def learning_rate(self):
        return self.thisptr.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.thisptr.learning_rate = value

cdef class SimpleSGDTrainer(Trainer):
    """Stochastic gradient descent trainer
    
    This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        learning_rate(number): Initial learning rate (default: 0.1)
    """
    def __cinit__(self, ParameterCollection m, float learning_rate = 0.1):
        self.thisptr = new CSimpleSGDTrainer(m.thisptr, learning_rate)
    def whoami(self):
        return "SimpleSGDTrainer"

cdef class CyclicalSGDTrainer(Trainer):
    """This trainer performs stochastic gradient descent with a cyclical learning rate as proposed in `Smith, 2015 <https://arxiv.org/abs/1506.01186>`_.

    This uses a triangular function with optional exponential decay.

    More specifically, at each update, the learning rate :math:`\eta` is updated according to :

    .. math::

        \\begin{split}
        \\text{cycle} &= \left\lfloor 1 + \\frac{\\texttt{it}}{2 \\times\\texttt{step_size}} \\right\\rfloor\\\\
       x &= \left\\vert \\frac{\\texttt{it}}{\\texttt{step_size}} - 2 \\times \\text{cycle} + 1\\right\\vert\\\\
       \eta &= \eta_{\\text{min}} + (\eta_{\\text{max}} - \eta_{\\text{min}}) \\times \max(0, 1 - x) \\times \gamma^{\\texttt{it}}\\\\
       \end{split}
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        learning_rate_min (number): Lower learning rate (default: {0.01})
        learning_rate_max (number): Upper learning rate (default: {0.1})
        step_size (number): Period of the triangular function in number of iterations (__not__ epochs). According to the original paper, this should be set around (2-8) x (training iterations in epoch) (default: {2000})
        gamma (number): Learning rate upper bound decay parameter (default: {0.0})
    """
    cdef CCyclicalSGDTrainer *thischildptr
    def __cinit__(self, ParameterCollection m, float learning_rate_min = 0.01, float learning_rate_max = 0.1, float step_size = 2000, float gamma = 0.0):
        self.thischildptr = self.thisptr = new CCyclicalSGDTrainer(m.thisptr, learning_rate_min, learning_rate_max, step_size, gamma)
    cpdef update(self):
        self.thischildptr.update()
    def whoami(self):
        return "CyclicalSGDTrainer"

cdef class MomentumSGDTrainer(Trainer):
    """Stochastic gradient descent with momentum
    
    This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory. 
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        learning_rate(number): Initial learning rate (default: 0.1)
        mom(number): Momentum (default: 0.9)

    """
    def __cinit__(self, ParameterCollection m, float learning_rate = 0.01, float mom = 0.9):
        self.thisptr = new CMomentumSGDTrainer(m.thisptr, learning_rate, mom)
    def whoami(self):
        return "MomentumSGDTrainer"


cdef class AdagradTrainer(Trainer):
    """Adagrad optimizer
    
    The adagrad algorithm assigns a different learning rate to each parameter.
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        learning_rate(number): Initial learning rate (default: 0.1)
        eps(number): Epsilon parameter to prevent numerical instability (default: 1e-20)
    """
    def __cinit__(self, ParameterCollection m, float learning_rate = 0.1, float eps = 1e-20):
        self.thisptr = new CAdagradTrainer(m.thisptr, learning_rate, eps)
    def whoami(self):
        return "AdagradTrainer"


cdef class AdadeltaTrainer(Trainer):
    """AdaDelta optimizer
    
    The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        eps(number): Epsilon parameter to prevent numerical instability (default: 1e-6)
        rho(number): Update parameter for the moving average of updates in the numerator (default: 0.95)
    """
    def __cinit__(self, ParameterCollection m, float eps = 1e-6, float rho = 0.95):
        self.thisptr = new CAdadeltaTrainer(m.thisptr, eps, rho)
    def whoami(self):
        return "AdadeltaTrainer"

cdef class RMSPropTrainer(Trainer):
    """RMSProp optimizer
    
    The RMSProp optimizer is a variant of Adagrad where the squared sum of previous gradients is replaced with a moving average with parameter rho.
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        learning_rate(number): Initial learning rate (default: 0.001)
        eps(number): Epsilon parameter to prevent numerical instability (default: 1e-8)
        rho(number): Update parameter for the moving average (`rho = 0` is equivalent to using Adagrad) (default: 0.9)
    """
    def __cinit__(self, ParameterCollection m, float learning_rate = 0.001,float eps = 1e-8, float rho = 0.9):
        self.thisptr = new CRMSPropTrainer(m.thisptr, learning_rate, eps, rho)
    def whoami(self):
        return "RMSPropTrainer"

cdef class AdamTrainer(Trainer):
    """Adam optimizer
    
    The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient
    
    Args:
        m(dynet.ParameterCollection): ParameterCollection to be trained
    
    Keyword Args:
        alpha(number): Initial learning rate (default: 0.001)
        beta_1(number): Moving average parameter for the mean (default: 0.9)
        beta_2(number): Moving average parameter for the variance (default: 0.999)
        eps(number): Epsilon parameter to prevent numerical instability (default: 1e-8)
    """
    def __cinit__(self, ParameterCollection m, float alpha = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8 ):
        self.thisptr = new CAdamTrainer(m.thisptr, alpha, beta_1, beta_2, eps)
    def whoami(self):
        return "AdamTrainer"

# Trainers }}}


# {{{ Softmax Builders
cdef class SoftmaxBuilder:
    """Interface for building softmax layers

    A softmax layer returns a probability distribution over :math:`C` classes given a vector :math:`h\in\mathbb R^d`, with

    .. math::
        p(c)\propto \exp(W_i^Th + b_i)\ \\forall i\in\{1\ldots C\}

    Where :math:`W\in \mathbb R^{C\\times d}, b \in \mathbb R^C`
    """

    cdef CSoftmaxBuilder *thisptr
    cdef int cg_version
    cdef int const_cg_version
    def __dealloc__(self):
        del self.thisptr

    cdef check_and_renew_graph(self, bool update):
        if update:
            if self.cg_version != cg_version():
                self.cg_version = cg_version()
                self.thisptr.new_graph(cg().thisptr[0], update)
        else:
            if self.const_cg_version != cg_version():
                self.const_cg_version = cg_version()
                self.thisptr.new_graph(cg().thisptr[0], update)

    cpdef neg_log_softmax(self, Expression x, unsigned c, bool update=True):
        """Negative log probability of a class
        
        Given class :math:`c` and vector :math:`x`, this returns :math:`-\log(p(c \mid x))`
        
        Args:
            x(dynet.Expression): Input vector
            c(unsigned): Class id
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Log probability of given class
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thisptr.neg_log_softmax(x.c(), c))

    cpdef neg_log_softmax_batch(self, Expression x, vector[unsigned] c, bool update=True):
        """Batched version of :code:`neg_log_softmax`
        
        Args:
            x(dynet.Expression): Input vector (batched)
            c(list): list of class ids (one per batch element)
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Log probability of given class
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thisptr.neg_log_softmax(x.c(), c))

    cpdef sample(self, Expression x):
        """Sample from the softmax distribution
        
        Args:
            x(dynet.Expression): Input vector
        
        Returns:
            Sampled class
            int
        """
        self.check_and_renew_graph(True)
        return self.thisptr.sample(x.c())

    cpdef full_log_distribution(self, Expression x, bool update=True):
        """Returns an Expression representing a vector the size of the number of classes.
        
        The ith dimension gives :math:`\log p(c_i | x)`. This function may be SLOW. Avoid if possible.
        
        Args:
            x(dynet.Expression): Input vector
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Vector of :math:`\log(p(c\mid x)`
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thisptr.full_log_distribution(x.c()))

    cpdef full_logits(self, Expression x, bool update=True):
        """Returns the logits (before application of the softmax)
        
        The ith dimension gives :math:`W_i^Tx + b_i`
        
        Args:
            x(dynet.Expression): Input vector
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Expression for the logits
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thisptr.full_logits(x.c()))

    cpdef ParameterCollection param_collection(self):
        """Returns the ParameterCollection containing the softmax parameters
        
        The first parameter in the parametercollection is the weight matrix, the second is the biases (if any)
        
        Returns:
            Subcollection holding the parameters
            ParameterCollection
        """
        return ParameterCollection.wrap(self.thisptr.get_parameter_collection())



cdef class StandardSoftmaxBuilder(SoftmaxBuilder):
    """
    This class implements the standard Softmax
    """

    def __cinit__(self, unsigned input_dim, unsigned num_classes, ParameterCollection pc, bool bias=True):
        """Constructs a softmaxbuilder

        Args:
            rep_dim(unsigned): Dimension of the input vectors
            num_classes(unsigned): Number of classes
            pc(dynet.ParameterCollection): Parameter collection
            bias(bool): Whether to use a bias vector or not
        """
        self.thisptr = new CStandardSoftmaxBuilder(input_dim, num_classes, pc.thisptr, bias)


cdef class ClassFactoredSoftmaxBuilder(SoftmaxBuilder):
    """Class factored softmax

    Each class is separated into a subclass, ie :math:`p(i\mid h)=p(i\mid h, c) p(c\mid h)` where :math:`c` is a class and :math:`i` a subclass

    """
    cdef CClassFactoredSoftmaxBuilder *thiscfptr
    cdef CDict cdic

    cdef dict_to_cdict(self, dict dic):
        words = sorted(dic.keys(), key=lambda x: dic[x])
        for w in words:
            i = self.cdic.convert(w.encode('utf8'))
            if i != dic[w]:
                raise ValueError('Dictionary should have unique ids from 0 to num_classes')
        self.cdic.freeze()

    def __cinit__(self, unsigned input_dim, str cluster_file, dict dic, ParameterCollection pc, bool bias=True):
        """Constructor from file

        This constructs the CFSM from a file with lines of the following format

        .. code::

            CLASSID   word    [freq]

        For words for instance

        Args:
            input_dim (unsigned): Dimension of the input vectors
            cluster_file (str): File containing classes
            dic (dict): A python dictionary converting words to indices. The dict should be one to one with :math:`\{0,\ldots,\\texttt{num_classes}-1\}` and cover all the words in the :code:`cluster_file`
            pc(dynet.ParameterCollection): Parameter collection
            bias(bool): Whether to use a bias vector or not
        """
        self.dict_to_cdict(dic)
        cdef string _fname = <string> cluster_file.encode("utf8")
        self.thiscfptr = self.thisptr = new CClassFactoredSoftmaxBuilder(input_dim, _fname, self.cdic, pc.thisptr, bias)


    cpdef class_log_distribution(self, Expression x, bool update=True):
        """Get log distribution over classes
        
        Args:
            x(dynet.Expression): Input vector
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Vector of :math:`\log(p(c\mid x)`
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thiscfptr.class_log_distribution(x.c()))

    cpdef class_logits(self, Expression x, bool update=True):
        """Returns the logits over classes
        
        Args:
            x(dynet.Expression): Input vector
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Expression for the logits
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thiscfptr.class_logits(x.c()))

    cpdef subclass_log_distribution(self, Expression x, unsigned classid, bool update=True):
        """ Get log distribution over subclasses of class
        
        Args:
            x(dynet.Expression): Input vector
            classid(int): class index
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Vector of :math:`\log(p(i\mid x, \\texttt{classid})`
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thiscfptr.subclass_log_distribution(x.c(), classid))

    cpdef subclass_logits(self, Expression x, unsigned classid, bool update=True):
        """Logits over subclasses of class
        
        Args:
            x(dynet.Expression): Input vector
            classid(int): class index
            update(bool): Whether to update the parameters or not (default: {True})
        
        Returns:
            Expression for the logits
            dynet.Expression
        """
        self.check_and_renew_graph(update)
        return Expression.from_cexpr(self.cg_version, self.thiscfptr.subclass_logits(x.c(), classid))

# Softmax Builders }}}
