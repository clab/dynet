# python dynet API changes for v2

* Model no longer holds named parameters
* checkpoint / revert mechanism for computation graph (useful for beam search etc)
* parameter initalization API

## Model no longer holds named parameters

The major API change in v2 of pydynet is in the `Model` class.
This change breaks backward compatibility, but is easy to adapt to.

The `Model` class no longer holds named parameters. This is done in order
to be more compatible to the C++ API, as well as to simplify the design of
the Model class. If a string-to-parameter mapping is desired, it can be achieved
externally to the model class.

Old API:
```python
m = Model()
m.add_parameters("w",(100,100))
m.add_lookup_parameters("lp",(100,100))

p_w = m["w"]
lp  = m["lp"]
```

New API:
```python
m = Model()
p_w = m.add_parameters((100,100))
lp = m.add_lookup_parameters((100,100))

# OR
params = {}
params["w"] = m.add_parameters((100,100))
params["lp"] = m.add_lookup_parameters((100,100))

p_w = params["w"]
lp  = params["lp"]
```


## New model-saving mechanism

There are now two model-saving mechaisms, "old" and "new", both work.

### The old mechanism

In the "old" mechanism (which still works) the user can load and save models using
a single call, provided that they added (either directly and indirectly) the exact same parameters
to the model prior to calling save and load.
For example:

```python

# saving:
from pydynet import *
m = Model()
W = m.add_parameters((100,100))
lb = LSTMBuilder(1, 100, 100, m) # this also adds parameters to the model
b = m.add_parameters((30))
m.save("filename")

# loading
m = Model()
W = m.add_parameters((100,100))
lb = LSTMBuilder(1, 100, 100, m) 
b = m.add_parameters((30))
m.load("filename")
```

Notice how the exact same parameters (same sizes and order) were added before calling "load".
This may be either a blessing or a curse.

## The new mechanism

In the new mechanism, the user instead tells the model, at save time, which are the components it is
interested in saving. They then need to specify the same components, in the same order, at load time.
Notice however that there is no need to specify the sizes etc, as this is handled by the save/load mechanism:

```python
# saving:
from pydynet import *
m = Model()
W = m.add_parameters((100,100))
lb = LSTMBuilder(1, 100, 100, m) # this also adds parameters to the model
b = m.add_parameters((30))
m.save("filename", [W,b,lb])

# loading
m = Model()
(W, b, lb) = m.load("filename")
```

Some may view this form of loading to be much more convenient.

In order to make use of "the new way", the items that are being passed in the list must adhere to at least one of the following:

* be of type `Parameters` or `LookupParameters` (the return types of `add_parameters` or `add_lookup_parameters`).
* be of a built-in "complex" builders such as `LSTMBuilder` or `GRUBuilder` that add parameters to the model.
* user defied classes that extend to the new `pydynet.Saveable` class and implement the required interface.


The `Saveable` class is used for easy creation of user-defined "sub networks" that can be saved and loaded as part of the model saving mechanism.

```python
class OneLayerMLP(Saveable):
    def __init__(self, model, num_input, num_hidde, num_out, act=tanh):
        self.W1 = model.add_parameters("W1", (num_hidden, num_input))
        self.W2 = model.add_parameters("W2", (num_out, num_hidden))
        self.b1 = model.add_parameters("b1", (num_hidden))
        self.b2 = model.add_parameters("b2", (num_out))
        self.act = act
        self.shape = (num_input, num_out)

    def __call__(self, input_exp):
        W1 = parameter(self.W1)
        W2 = parameter(self.W2)
        b1 = parameter(self.b1)
        b2 = parameter(self.b2)
        g = self.act
        return softmax(W2*g(W1*input_exp + b1)+b2)

    # the Saveable interface requires the implementation
    # of the two following methods, specifying all the 
    # Parameters / LookupParameters / LSTMBuilder / Saveables / etc 
    # that are directly created by this Saveable.
    def get_components(self):
        return (self.W1, self.W2, self.b1, self.b2)
            
    def restore_components(self, components):
        self.W1, self.W2, self.b1, self.b2 = components

```

And for the usage:
```python

m = Model()
# create an embedding table.
E = m.add_lookup_parameters((1000,10))
# create an MLP from 10 to 4 with a hidden layer of 20.
mlp = OneLayerMLP(m, 10, 20, 4, rectify)

# use them together.
output = mlp(E[3])

# now save the model:
m.save("filename",[mlp, E])

# now load:
m2 = Model()
mlp2, E2 = m.load("filename")

output2 = mlp2(E2[3])

assert(numpy.array_equal(output2.npvalue(), output.npvalue()))
```

## Checkpoint / revert mechanism for computation graph

When doing beam search, we often do multiple calculations and then discard most of them.
If the calculations are done with the compuation graph, the computation graph can grow very
large (and hence become slow and memory consuming) as the expressions resulting from the discarded
computation are not discarded from the graph.

The new checkpointing mechanism deals with this problem by allowing to mark certain points in the graph lifestage, and then returning back to them (deleting everything that was created after the checkpoint).

The API is using `cg_checkpoint()` to mark a checkpoint, and `cg_revert()` to return to the last checkpoint.
The checkpoints are treated as a stack, so you can create several checkpoints and then return to them
in reverse order.

Be careful with this feature, as expressions that were created after the checkpoint will be invaludated after the revert, but this is not enforced in code so accessing them _may work_, but result in wrong computations.

Example usage:
```python

m = Model();
px = m.add_parameters((10,10))
x = parameter(px)
y = x*x
cg_checkpoint()
z = y+y
w = z+y
print w.npvalue()
cg_revert()
# at this point x and y are still alive, but z and w are deleted,
# they are not part of the computation graph anymore and the c-level memory
# for them is freed.

```

## Parameter Initialization

As before, parameters are created with:
```python
import dynet as dy
m = dy.Model()
dim = (10,10) # either a 2-dim tuple or a scaler.
p = m.add_parameters(dim)
```

This will initialize the parameters according to Glorot Initialization.

Other initializations can be specified by passing an initializer object:

```python
import dynet as dy
m = dy.Model()
dim = (10,10)
p = m.add_parameters(dim,init=GlorotInitializer()) 
```

Possible initializers are:
```python
init1 = dy.NormalInitializer(mean = 0, var = 1) # normal with mean and variance.
init2 = dy.UniformInitializer(scale)  # uniform between -scale and scale.
init3 = dy.ConstInitializer(c)  # all values are c
init4 = dy.GlorotInitializer()
```

Parameters can also be initialized from arbitraty numpy arrays:

```python
p1 = m.parameters_from_numpy(np.eye(10))
p2 = m.parameters_from_numpy(np.array([[1,2,3],[4,5,6]]))
```



