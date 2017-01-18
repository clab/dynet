Basic Tutorial
~~~~~~~~~~~~~~

An illustration of how models are trained (for a simple logistic
regression model) is below:

 First, we set up the structure of the model.

Create a model, and an SGD trainer to update its parameters.

.. code:: cpp

    Model mod;
    SimpleSGDTrainer sgd(mod);

Create a "computation graph," which will define the flow of information.

.. code:: cpp

    ComputationGraph cg;

Initialize a 1x3 parameter vector, and add the parameters to be part of the computation graph.

.. code:: cpp

    Expression W = parameter(cg, mod.add_parameters({1, 3}));

Create variables defining the input and output of the regression, and load them into the computation graph. Note that we don't need to set concrete values yet.

.. code:: cpp

    vector<dynet::real> x_values(3);
    Expression x = input(cg, {3}, &x_values);
    dynet::real y_value;
    Expression y = input(cg, &y_value);

Next, set up the structure to multiply the input by the weight vector,  then run the output of this through a logistic sigmoid function logistic regression).

.. code:: cpp

    Expression y_pred = logistic(W*x);

Finally, we create a function to calculate the loss. The model will be optimized to minimize the value of the final function in the computation graph.

.. code:: cpp

    Expression l = binary_log_loss(y_pred, y);

We are now done setting up the graph, and we can print out its structure:

.. code:: cpp

    cg.print_graphviz();

Now, we perform a parameter update for a single example. Set the input/output to the values specified by the training data:

.. code:: cpp

    x_values = {0.5, 0.3, 0.7};
    y_value = 1.0;

"forward" propagates values forward through the computation graph, and returns the loss.

.. code:: cpp

    dynet::real loss = as_scalar(cg.forward(l));

"backward" performs back-propagation, and accumulates the gradients of the parameters within the "Model" data structure.

.. code:: cpp

    cg.backward(l);

"sgd.update" updates parameters of the model that was passed to its constructor. Here 1.0 is the scaling factor that allows us to control the size of the update.

.. code:: cpp

    sgd.update(1.0);

Note that this very simple example that doesn't cover things like memory
initialization, reading/writing models, recurrent/LSTM networks, or
adding biases to functions. The best way to get an idea of how to use
DyNet for real is to look in the ``example`` directory, particularly
starting with the simplest ``xor`` example.