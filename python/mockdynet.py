class DynetParams:
    """This object holds the global parameters of Dynet

    You should only need to use this after importing dynet as :

        import _dynet / import _gdynet

    See the documentation for more details
    """
    def __init__(self):
        pass
    def from_args(self):
        """Gets parameters from the command line arguments
        
        You can still modify the parameters after calling this.
        See the documentation about command line arguments for more details
        
        Keyword Args:
            shared_parameters([type]): [description] (default: (None))
        """
        pass
    def init(self):
        """Initialize dynet with the current dynetparams object.
        
        This is one way, you can't uninitialize dynet
        """
        pass
    def set_mem(self):
        """Set the memory allocated to dynet
        
        The unit is MB
        
        Args:
            mem(number): memory size in MB
        """
        pass
    def set_random_seed(self):
        """Set random seed for dynet
        
        Args:
            random_seed(number): Random seed
        """
        pass
    def set_weight_decay(self):
        """Set weight decay parameter
        
        Args:
            weight_decay(float): weight decay parameter
        """
        pass
    def set_shared_parameters(self):
        """Shared parameters
        
        Args:
            shared_parameters(bool): shared parameters
        """
        pass
    def set_requested_gpus(self):
        """Number of requested gpus
        
        Currently only 1 is supported
        
        Args:
            requested_gpus(number): number of requested gpus
        """
        pass
    def set_gpu_mask(self):
        pass
def init():
    """Initialize dynet
    
    Initializes dynet from command line arguments. Do not use after 
        
        import dynet

    only after 

        import _dynet / import _gdynet
    
    Keyword Args:
        shared_parameters(bool): [description] (default: (None))
    """
    pass
def init_from_params():
    """Initialize from DynetParams
    
    Same as 

        params.init()
    
    Args:
        params(DynetParams): dynet parameters
    """
    pass
    """Get dynet Dim from tuple
    

    Args:
        dim(tuple): Dimensions as a tuple
        batch_size(number): Batch size (default: (1))
    
    Returns:
        CDim: Dynet dimension
    """
    """
    Returns a tuple (dims,batch_dim) where dims is the tuple of dimensions of each batch element
    """
class Parameters:
    """Parameters class
    
    Parameters are things that are optimized. in contrast to a system like Torch where computational modules may have their own parameters, in DyNet parameters are just parameters.
    """
    def __cinit__(self):
        pass
    def shape(self):
        """[summary]
        
        [description]
        
        Returns:
            [type]: [description]
        """
        pass
    def as_array(self):
        """Return as a numpy array.
        
        Returns:
            np.ndarray: values of the parameter
        """
        pass
    def grad_as_array(self):
        """Return gradient as a numpy array.
        
        Returns:
            np.ndarray: values of the gradient w.r.t. this parameter
        """
        pass
    def clip_inplace(self):
        """Clip the values in the parameter to a fixed range [left, right] (in place)
        
        Returns:
            None
        """
        pass
    def load_array(self):
        """Deprecated
        """
        pass
    def zero(self):
        """Set the parameter to zero

        """
        pass
    def scale(self):
        """Scales the parameter

        Args:
            s(float): Scale

        """
        pass
    def is_updated(self):
        """check whether the parameter is updated or not
        
        Returns:
            bool: Update status
        """
        pass
    def set_updated(self):
        """Set parameter as "updated"
        
        Args:
            b(bool): updated status
        """
        pass
    def get_index(self):
        """Get parameter index
        
        Returns:
            unsigned: Index of the parameter
        """
        pass
    def expr(self):
        """Returns the parameter as an expression

        This is the same as calling

            dy.parameter(param)
        
        Args:
            update(bool): If this is set to False, the parameter won't be updated during the backward pass
        Returns:
            Expression: Expression of the parameter
        """
        pass
class LookupParameters:
    def __cinit__(self):
        pass
    def init_from_array(self):
        pass
    def shape(self):
        pass
    def __getitem__(self):
        pass
    def batch(self):
        pass
    def init_row(self):
        pass
    def as_array(self):
        """
        Return as a numpy array.
        """
        pass
    def grad_as_array(self):
        """
        Return gradients as a numpy array.
        """
        pass
    def scale(self):
        """Scales the parameter

        Args:
            s(float): Scale

        """
        pass
    def expr(self):
        pass
    def zero(self):
        pass
    def is_updated(self):
        pass
    def set_updated(self):
        pass
    def get_index(self):
        pass
    def __init__(self):
        pass
    def __getstate__(self):
        pass
    def get_components(self):
        """
        List of parameter-containing components that are
        members of this object and are created by it.
        """
        pass
    def restore_components(self):
        pass
class PyInitializer:
    def __init__(self):
        pass
    def __dealloc__(self):
        pass
class NormalInitializer(PyInitializer):
    def __init__(self):
        pass
class UniformInitializer(PyInitializer):
    def __init__(self):
        pass
class ConstInitializer(PyInitializer):
    def __init__(self):
        pass
class IdentityInitializer(PyInitializer):
    def __init__(self):
        pass
class GlorotInitializer(PyInitializer):
    def __init__(self):
        pass
class SaxeInitializer(PyInitializer):
   def __init__(self):
       pass
class FromFileInitializer(PyInitializer):
    def __init__(self):
        pass
class NumpyInitializer(PyInitializer):
    def __init__(self):
        pass
class Model:
    def __cinit__(self):
        pass
    def __init__(self):
        pass
    def __dealloc__(self):
        pass
    def from_file():
        pass
    def pl(self):
        pass
    def parameters_from_numpy(self):
        pass
    def add_parameters(self):
        pass
    def add_lookup_parameters(self):
        pass
    def save_all(self):
        pass
    def load_all(self):
        pass
    def save(self):
        pass
    def load(self):
        pass
class UnsignedValue:
    def __cinit__(self):
        pass
    def set(self):
        pass
    def get(self):
        pass
class FloatValue:
    def __cinit__(self):
        pass
    def set(self):
        pass
    def get(self):
        pass
class UnsignedVectorValue:
    def __cinit__(self):
        pass
    def __dealloc__(self):
        pass
    def set(self):
        pass
    def get(self):
        pass
    def size(self):
        pass
class FloatVectorValue:
    def __cinit__(self):
        pass
    def __dealloc__(self):
        pass
    def set(self):
        pass
    def get(self):
        pass
    def size(self):
        pass
def cg_version():
    pass
def renew_cg():
    pass
def print_text_graphviz():
    pass
def cg_checkpoint():
    pass
def cg_revert():
    pass
def cg():
    pass
class ComputationGraph:
    def __cinit__(self):
        pass
    def __dealloc__(self):
        pass
    def renew(self):
        pass
    def version(self):
        pass
    def parameters(self):
        pass
    def forward_scalar(self):
        pass
    def inc_forward_scalar(self):
        pass
    def forward_vec(self):
        pass
    def inc_forward_vec(self):
        pass
    def forward(self):
        pass
    def inc_forward(self):
        pass
    def backward(self):
        pass
    def print_graphviz(self):
        pass
    def checkpoint(self):
        pass
    def revert(self):
        pass
    def inputMatrixLiteral(self):
        pass
class Expression:
    """Expressions are the building block of a Dynet computation graph.
    
    Expressions are the main data types being manipulated in a DyNet program. Each expression represents a sub-computation in a computation graph.
    """
    def __cinit__(self):
        pass
    def dim(self):
        """Dimension of the expression

        Returns a tuple (dims,batch_dim) where dims is the tuple of dimensions of each batch element
        
        Returns:
            tuple: dimension
        """
        pass
    def __repr__(self):
        pass
    def __str__(self):
        """Returns a string representation of the expression
        
        The format is "expression [expression id]/[computation graph id]"
        """
        pass
    def __getitem__(self):
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
        pass
    def scalar_value(self):
        """Returns value of an expression as a scalar
        
        This only works if the expression is a scalar
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: (False))
        
        Returns:
            float: Scalar value of the expression
        """
        pass
    def vec_value(self):
        """Returns the value of the expression as a vector
        
        In case of a multidimensional expression, the values are flattened according to a column major ordering
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: (False))
        
        Returns:
            list: Array of values
        """
        pass
    def npvalue(self):
        """Returns the value of the expression as a numpy array
        
        The last dimension is the batch size (if it's > 1)
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: (False))
        
        Returns:
            np.ndarray: numpy array of values
        """
        pass
    def value(self):
        """Gets the value of the expression in the most relevant format
        
        this returns the same thing as `scalar_value`, `vec_value`, `npvalue` depending on whether the number of dimensions of the expression is 0, 1 or 2+
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: (False))
        
        Returns:
            float, list, np.ndarray: Value of the expression
        """
        pass
    def forward(self):
        """This runs incremental forward on the entire graph
        
        May not be optimal in terms of efficiency.
        Prefer `values`
        
        Keyword Args:
            recalculate(bool): Recalculate the computation graph (for static graphs with new inputs) (default: (False))
        """
        pass
    def backward(self):
        """Run the backward pass based on this expression
        
        The expression should be a scalar (objective)
        """
        pass
    def __add__(self):
        pass
    def __mul__(self):
        pass
    def __div__(self):
        pass
    def __truediv__(self):
        pass
    def __neg__(self):
        pass
    def __sub__(self):
        pass
def parameter():
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
    pass
class _inputExpression(Expression):
    """Subclass of Expression corresponding to scalar input expressions
    
    """
    def __cinit__(self):
        pass
    def set(self):
        """Change the value of the expression
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            s(float): New value
        """
        pass
def scalarInput():
    pass
class _vecInputExpression(Expression):
    """Subclass of Expression corresponding to any non-scalar input expressions
    
    Despite the name, this also represents tensors (in column major format).
    TODO : change this
    """
    def __cinit__(self):
        pass
    def set(self):
        """Change the value of the expression
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        For now this only accepts new values as flattened arrays (column majors). TODO : change this

        Args:
            data(vector[float]): New value
        """
        pass
def vecInput():
    """Input an empty vector
    
    Args:
        dim(number): Size
    
    Returns:
        _vecInputExpression: Corresponding expression
    """
    pass
def inputVector():
    """Input a vector by values
    
    Args:
        v(vector[float]): Values
    
    Returns:
        _vecInputExpression: Corresponding expression
    """
    pass
def matInput():
    """DEPRECATED : use inputTensor
    
    TODO : remove this
    
    Args:
        int d1([type]): [description]
        int d2([type]): [description]
    
    Returns:
        [type]: [description]
    """
    pass
def inputMatrix():
    """DEPRECATED : use inputTensor

    TODO : remove this

    inputMatrix(vector[float] v, tuple d)

    Create a matrix literal.
    First argument is a list of floats (or a flat numpy array).
    Second argument is a dimension.
    Returns: an expression.
    Usage example:

        x = inputMatrix([1,2,3,4,5,6],(2,3))
        x.npvalue()
        --> 
        array([[ 1.,  3.,  5.],
               [ 2.,  4.,  6.]])
    """
    pass
def inputTensor():
    """Creates a tensor expression based on a numpy array or a list.
    
    The dimension is inferred from the shape of the input.
    if batched=True, the last dimension is used as a batch dimension
    if arr is a list of numpy ndarrays, this returns a batched expression where the batch elements are the elements of the list
    
    Args:
        arr(list,np.ndarray): Values : numpy ndarray OR list of np.ndarray OR multidimensional list of floats
    
    Keyword Args:
        batched(bool): Whether to use the last dimension as a batch dimension (default: (False))
    
    Returns:
        _vecInputExpression: Input expression
    
    Raises:
        TypeError: If the type is not respected
    """
    pass
class _lookupExpression(Expression):
    """Expression corresponding to a lookup from lookup parameter
    
    """
    def __cinit__(self):
        pass
    def set(self):
        """Change the lookup index
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(number): New lookup index
        """
        pass
class _lookupBatchExpression(Expression):
    """Expression corresponding to batched lookups from a lookup parameter
    
    """
    def __cinit__(self):
        pass
    def set(self):
        """Change the lookup index
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(list(int)): New indices
        """
        pass
def lookup():
    """Pick an embedding from a lookup parameter and returns it as a expression

        :param p: Lookup parameter to pick from
        :type p: LookupParameters
    
    Keyword Args:
        index(number): Lookup index (default: (0))
        update(bool): Whether to update the lookup parameter [(default: (True))
    
    Returns:
        _lookupExpression: Expression for the embedding
    """
    pass
def lookup_batch():
    """Look up parameters.

    The mini-batched version of lookup. The resulting expression will be a mini-batch of parameters, where the "i"th element of the batch corresponds to the parameters at the position specified by the "i"th element of "indices"
    
    Args:
        p(LookupParameters): Lookup parameter to pick from
        indices(list(int)): Indices to look up for each batch element
    
    Keyword Args:
        update(bool): Whether to update the lookup parameter (default: (True))
    
    Returns:
        _lookupBatchExpression: Expression for the batched embeddings
    """
    pass
class _pickerExpression(Expression):
    """Expression corresponding to a row picked from a bigger expression
    
    """
    def __cinit__(self):
        pass
    def set_index(self):
        """Change the pick index
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(number): New index
        """
        pass
def pick():
    """Pick element.

    Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.
    
    Args:
        e(Expression): Expression to pick from
    
    Keyword Args:
        index(number): Index to pick (default: (0))
        dim(number): Dimension to pick from (default: (0))
    
    Returns:
        _pickerExpression: Picked expression
    """
    pass
class _pickerBatchExpression(Expression):
    """Batched version of `_pickerExpression`
    
    """
    def __cinit__(self):
        pass
    def set_index(self):
        """Change the pick indices
        
        This is useful if you want to to change the input and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(list): New list of indices
        """
        pass
def pick_batch():
    """Batched pick.

    Pick elements from multiple batches.
    
    Args:
        e(Expression): Expression to pick from
        indices(list): Indices to pick
        dim(number): Dimension to pick from (default: (0))
    
    Returns:
        _pickerBatchExpression: Picked expression
    """
    pass
class _hingeExpression(Expression):
    """Expression representing the output of the hinge operation
    
    """
    def __cinit__(self):
        pass
    def set_index(self):
        """Change the correct candidate index
        
        This is useful if you want to to change the target and recompute the graph without needing to re-create it. Don't forget to use `recalculate=True` when calling `.value()` on the output.
        This allows you to use dynet as a static framework.
        
        Args:
            i(number): New correct index
        """
        pass
def hinge():
    """Hinge loss.

    This expression calculates the hinge loss, formally expressed as: 
    
    Args:
        x(Expression): A vector of scores
        index(number): The index of the correct candidate
    
    Keyword Args:
        m(number): Margin (default: (1.0))
    
    Returns:
        _hingeExpression: The hinge loss of candidate index with respect to margin m
    """
    pass
def zeroes():
    pass
def random_normal():
    pass
def random_bernoulli():
    pass
def random_uniform():
    pass
def nobackprop():
    pass
def flip_gradient():
    pass
def cdiv():
    pass
def cmult():
    pass
def colwise_add():
    pass
def inverse():
    pass
def logdet():
    pass
def trace_of_product():
    pass
def dot_product():
    pass
def squared_norm():
    pass
def squared_distance():
    pass
def l1_distance():
    pass
def binary_log_loss():
    pass
def conv1d_narrow():
    pass
def conv1d_wide():
    pass
def filter1d_narrow():
    pass
def tanh():
    pass
def exp():
    pass
def square():
    pass
def sqrt():
    pass
def erf():
    pass
def cube():
    pass
def log():
    pass
def lgamma():
    pass
def logistic():
    pass
def rectify():
    pass
def log_softmax():
    pass
def softmax():
    pass
def sparsemax():
    pass
def softsign():
    pass
def pow():
    pass
def bmin():
    pass
def bmax():
    pass
def transpose():
    pass
def select_rows():
    pass
def select_cols():
    pass
def sum_cols():
    pass
def sum_elems():
    pass
def sum_batches():
    pass
def fold_rows():
    pass
def pairwise_rank_loss():
    pass
def poisson_loss():
    pass
def huber_distance():
    pass
def kmax_pooling():
    pass
def pickneglogsoftmax():
    pass
def pickneglogsoftmax_batch():
    pass
def kmh_ngram():
    pass
def pickrange():
    pass
def pick_batch_elem():
    pass
def pick_batch_elems():
    pass
def noise():
    pass
def dropout():
    pass
def block_dropout():
    pass
def reshape():
    pass
def esum():
    pass
def logsumexp():
    pass
def average():
    pass
def emax():
    pass
def concatenate_cols():
    pass
def concatenate():
    pass
def concat_to_batch():
    pass
def affine_transform():
    pass
class _RNNBuilder:
    def __dealloc__(self):
        pass
    def set_dropout(self):
        pass
    def disable_dropout(self):
        pass
    def initial_state(self):
        pass
    def initial_state_from_raw_vectors(self):
        pass
class SimpleRNNBuilder(_RNNBuilder):
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class GRUBuilder(_RNNBuilder):
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class LSTMBuilder(_RNNBuilder):
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class VanillaLSTMBuilder(_RNNBuilder):
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class FastLSTMBuilder(_RNNBuilder):
    def __cinit__(self):
        pass
    def whoami(self):
        pass
    """
    Builder for BiRNNs that delegates to regular RNNs and wires them together.  
    
        builder = BiRNNBuilder(1, 128, 100, model, LSTMBuilder)
        [o1,o2,o3] = builder.transduce([i1,i2,i3])
    """
    def __init__(self):
        """
        @param num_layers: depth of the BiRNN
        @param input_dim: size of the inputs
        @param hidden_dim: size of the outputs (and intermediate layer representations)
        @param model
        @param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
        @param builder_layers: list of (forward, backward) pairs of RNNBuilder instances to directly initialize layers
        """
        pass
    def whoami(self):
        pass
    def set_dropout(self):
        pass
    def disable_dropout(self):
        pass
    def add_inputs(self):
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
        pass
    def transduce(self):
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
        pass
class RNNState:
    """
    This is the main class for working with RNNs / LSTMs / GRUs.
    Request an RNNState initial_state() from a builder, and then progress from there.
    """
    def __cinit__(self):
        pass
    def set_h(self):
        pass
    def set_s(self):
        pass
    def add_input(self):
        pass
    def add_inputs(self):
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
        pass
    def transduce(self):
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
        pass
    def output(self):
        pass
    def h(self):
        """
        tuple of expressions representing the output of each hidden layer
        of the current step.
        the actual output of the network is at h()[-1].
        """
        pass
    def s(self):
        """
        tuple of expressions representing the hidden state of the current
        step.

        For SimpleRNN, s() is the same as h()
        For LSTM, s() is a series of of memory vectors, followed the series
                  followed by the series returned by h().
        """
        pass
    def prev(self):
        pass
    def b(self):
        pass
class StackedRNNState:
    def __init__(self):
        pass
    def add_input(self):
        pass
    def output(self):
        pass
    def prev(self):
        pass
    def h(self):
        pass
    def s(self):
        pass
    def add_inputs(self):
        """
        returns the list of states obtained by adding the given inputs
        to the current state, one by one.
        """
        pass
class Trainer:
    """
    Generic trainer
    """
    def __dealloc__(self):
        pass
    def update(self):
        """Update the parameters
        
        The update equation is different for each trainer, check the online c++ documentation for more details on what each trainer does
        
        Keyword Args:
            s(number): Optional scaling factor to apply on the gradient. (default: (1.0))
        """
        pass
    def update_subset(self):
        """Update a subset of parameters
        
        Only use this in last resort, a more elegant way to update only a subset of parameters is to use the "update" keyword in dy.parameter or Parameter.expr() to specify which parameters need to be updated __during the creation of the computation graph__
        
        Args:
            updated_params(list): Indices of parameters to update
            updated_lookups(list): Indices of lookup parameters to update
        
        Keyword Args:
            s(number): Optional scaling factor to apply on the gradient. (default: (1.0))
        """
        pass
    def update_epoch(self):
        """Update trainers hyper-parameters that depend on epochs
        
        Basically learning rate decay.
        
        Keyword Args:
            r(number): Number of epoch that passed (default: (1.0))
        """
        pass
    def status(self):
        """Outputs information about the trainer in the stderr 
        
        (number of updates since last call, number of clipped gradients, learning rate, etc...)
        """
        pass
    def set_sparse_updates(self):
        """Sets updates to sparse updates

        DyNet trainers support two types of updates for lookup parameters, sparse and dense. Sparse updates are the default. They have the potential to be faster, as they only touch the parameters that have non-zero gradients. However, they may not always be faster (particulary on GPU with mini-batch training), and are not precisely numerically correct for some update rules such as MomentumTrainer and AdamTrainer. Thus, if you set this variable to false, the trainer will perform dense updates and be precisely correct, and maybe faster sometimes.
        Args:
            su(bool): flag to activate/deactivate sparse updates
        """
        pass
    def set_clip_threshold(self):
        """Set clipping thershold
        
        To deactivate clipping, set the threshold to be <=0
        
        Args:
            thr(number): Clipping threshold
        """
        pass
    def get_clip_threshold(self):
        """Get clipping threshold
        
        Returns:
            number: Gradient clipping threshold
        """
        pass
class SimpleSGDTrainer(Trainer):
    """Stochastic gradient descent trainer
    
    This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.
    
    Args:
        m(dynet.Model): Model to be trained
    
    Keyword Args:
        e0(number): Initial learning rate (default: (0.1))
        edecay(number): Learning rate decay parameter (default: (0.0))
    """
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class MomentumSGDTrainer(Trainer):
    """Stochastic gradient descent with momentum
    
    This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory. 
    
    Args:
        m(dynet.Model): Model to be trained
    
    Keyword Args:
        e0(number): Initial learning rate (default: (0.1))
        mom(number): Momentum (default: (0.9))
        edecay(number): Learning rate decay parameter (default: (0.0))

    """
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class AdagradTrainer(Trainer):
    """Adagrad optimizer
    
    The adagrad algorithm assigns a different learning rate to each parameter.
    
    Args:
        m(dynet.Model): Model to be trained
    
    Keyword Args:
        e0(number): Initial learning rate (default: (0.1))
        eps(number): Epsilon parameter to prevent numerical instability (default: (1e-20))
        edecay(number): Learning rate decay parameter (default: (0.0))
    """
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class AdadeltaTrainer(Trainer):
    """AdaDelta optimizer
    
    The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.
    
    Args:
        m(dynet.Model): Model to be trained
    
    Keyword Args:
        eps(number): Epsilon parameter to prevent numerical instability (default: (1e-6))
        rho(number): Update parameter for the moving average of updates in the numerator (default: (0.95))
        edecay(number): Learning rate decay parameter (default: (0.0))
    """
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class RMSPropTrainer(Trainer):
    """RMSProp optimizer
    
    The RMSProp optimizer is a variant of Adagrad where the squared sum of previous gradients is replaced with a moving average with parameter rho.
    
    Args:
        m(dynet.Model): Model to be trained
    
    Keyword Args:
        e0(number): Initial learning rate (default: (0.001))
        eps(number): Epsilon parameter to prevent numerical instability (default: (1e-8))
        rho(number): Update parameter for the moving average (`rho = 0` is equivalent to using Adagrad) (default: (0.9))
        edecay(number): Learning rate decay parameter (default: (0.0))
    """
    def __cinit__(self):
        pass
    def whoami(self):
        pass
class AdamTrainer(Trainer):
    """Adam optimizer
    
    The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient
    
    Args:
        m(dynet.Model): Model to be trained
    
    Keyword Args:
        alpha(number): Initial learning rate (default: (0.001))
        beta_1(number): Moving average parameter for the mean (default: (0.9))
        beta_2(number): Moving average parameter for the variance (default: (0.999))
        eps(number): Epsilon parameter to prevent numerical instability (default: (1e-8))
        edecay(number): Learning rate decay parameter (default: (0.0))
    """
    def __cinit__(self):
        pass
    def whoami(self):
        pass
