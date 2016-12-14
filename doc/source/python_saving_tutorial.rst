.. role:: python(code)
   :language: python

Saving Models
~~~~~~~~~~~~~

In order to save model parameters, the user instead tells the model, at save time, which are the components it is
interested in saving. They then need to specify the same components, in the same order, at load time.
Notice however that there is no need to specify the sizes etc, as this is handled by the save/load mechanism:

.. code:: python

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

The items that are being passed in the list must adhere to at least one of the following:

* be of type :python:`Parameters` or :python:`LookupParameters` (the return types of :python:`add_parameters` or :python:`add_lookup_parameters`).
* be of a built-in "complex" builders such as :python:`LSTMBuilder` or :python:`GRUBuilder` that add parameters to the model.
* user defined classes that extend to the new :python:`pydynet.Saveable` class and implement the required interface.


The :python:`Saveable` class is used for easy creation of user-defined "sub networks" that can be saved and loaded as part of the model saving mechanism.

.. code:: python

  class OneLayerMLP(Saveable):
      def __init__(self, model, num_input, num_hidden, num_out, act=tanh):
          self.W1 = model.add_parameters((num_hidden, num_input))
          self.W2 = model.add_parameters((num_out, num_hidden))
          self.b1 = model.add_parameters((num_hidden))
          self.b2 = model.add_parameters((num_out))
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


And for the usage:

.. code:: python

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