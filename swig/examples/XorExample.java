import edu.cmu.dynet.*;
import static edu.cmu.dynet.dynet_swig.*;

// Simple example training an xor classifier for java binding of dynet (following xor.cc example)
public class XorExample {

  static final int HIDDEN_SIZE = 8;
  static final int ITERATIONS = 30;

  // Temporary convenience function (in lieu of future library support)
  static Dim makeDim(int[] dims) {
    LongVector dimInts = new LongVector();
    for (int i = 0; i < dims.length; i++) {
      dimInts.add(dims[i]);
    }
    return new Dim(dimInts);
  }

  public static void main(String[] args) {
    System.out.println("Running XOR example");
    myInitialize();
    System.out.println("Dynet initialized!");
    Model m = new Model();
    SimpleSGDTrainer sgd = new SimpleSGDTrainer(m);
    ComputationGraph cg = new ComputationGraph();

    // Declare parameters
    Parameter p_W = m.add_parameters(makeDim(new int[]{HIDDEN_SIZE, 2}));
    Parameter p_b = m.add_parameters(makeDim(new int[]{HIDDEN_SIZE}));
    Parameter p_V = m.add_parameters(makeDim(new int[]{1, HIDDEN_SIZE}));
    Parameter p_a = m.add_parameters(makeDim(new int[]{1}));

    // Bind parameters to compute graph expressions
    Expression W = parameter(cg, p_W);
    Expression b = parameter(cg, p_b);
    Expression V = parameter(cg, p_V);
    Expression a = parameter(cg, p_a);

    // Clunky way of setting up 2-element input vector
    FloatVector x_values = new FloatVector();
    x_values.add(0.0f);
    x_values.add(0.0f);

    // Declare as input node in computation graph
    Expression x = input(cg, makeDim(new int[]{2}), x_values);

    // Output value
    float y_value = 0f;
    Expression y = input(cg, y_value);

    // Hidden perceptron layer
    Expression h = tanh(exprPlus(exprTimes(W, x), b));

    // Predicted output
    Expression y_pred = exprPlus(exprTimes(V, h), a);

    // Loss calculation
    Expression loss_expr = squared_distance(y_pred, y);

    // Print graph structure (can be fed to graphviz)
    System.out.println();
    System.out.println("Computation graphviz structure:");
    cg.print_graphviz();

    // Train the parameters
    System.out.println();
    System.out.println("Training...");
    for (int iter = 0; iter < ITERATIONS; iter++) {
      float loss = 0;
      for (int mi = 0; mi < 4; mi++) {
        boolean x1 = mi % 2 > 0;
        boolean x2 = (mi / 2) % 2 > 0;
        x_values.set(0, x1 ? 1 : -1);
        x_values.set(1, x2 ? 1 : -1);
        y_value = (x1 != x2) ? 1 : -1;
        loss += as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        sgd.update(1.0f);
      }
      sgd.update_epoch();
      loss /= 4;
      System.out.println("iter = " + iter + ", loss = " + loss);
    }
  }

  // TODO: Figure out serialization

}