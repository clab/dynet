using System;
using System.Linq;
using dynetsharp;
using dy = dynetsharp.DynetFunctions;

namespace DynetSharpExamples {
    class Program {
        static void Main(string[] args) {
            DynetParams.FromArgs(args).Initialize();      
            // Alternatively, can initialize and it directly, e.g:
            // DynetParams dp = new DynetParams();
            // dp.AutoBatch = true;
            // dp.MemDescriptor = "768";
            // dp.Initialize();

            const int ITERATIONS = 30;
            const int HIDDEN_SIZE = 8;

            // ParameterCollection (all the model parameters).
            ParameterCollection m = new ParameterCollection();
            Trainer trainer = new SimpleSGDTrainer(m);

            // Create the parameters
            Parameter p_W = m.AddParameters(new[] { HIDDEN_SIZE, 2 });
            Parameter p_b = m.AddParameters(new[] { HIDDEN_SIZE });
            Parameter p_V = m.AddParameters(new[] { 1, HIDDEN_SIZE });
            Parameter p_a = m.AddParameters(new[] { 1 });

            // Load the model?
            string modelFname = args.FirstOrDefault(arg => arg.StartsWith("-model="));
            if (modelFname != null) {
                modelFname = modelFname.Substring("-model=".Length);
                m.Load(modelFname);
            }

            // For good practice, renew the computation graph
            dy.RenewCG();

            /*/ In the past, we need to explicitly convert Parameters to Expressions, now
            // it's all done automatically.
            // Build the graph
            Expression W = dy.parameter(p_W); // Can also do: p_W.ToExpression();
            Expression b = dy.parameter(p_b);
            Expression V = dy.parameter(p_V);
            Expression a = dy.parameter(p_a); */

            // Set x_values to change the inputs to the network.
            Expression x = dy.inputVector(2);
            // Set y_value to change the target output
            Expression y = dy.input(0f);

            Expression h = dy.tanh(p_W * x + p_b);
            Expression y_pred = p_V * h + p_a;
            Expression loss_expr = dy.squared_distance(y_pred, y);

            // Show the computation graph, just for fun.
            dy.PrintCGGraphViz();

            // Train the parameters.
            for (int iIter = 0; iIter < ITERATIONS; iIter++) {
                double loss = 0;
                for (int mi = 0; mi < 4; mi++) {
                    float x1 = (mi % 2) != 0 ? 1 : -1;
                    float x2 = ((mi / 2) % 2) != 0 ? 1 : -1;
                    float yValue = (x1 != x2) ? 1 : -1;
                    // Set the values
                    x.SetValue(new[] { x1, x2 });
                    y.SetValue(yValue);
                    // Forward & backward
                    loss += loss_expr.ScalarValue(fRecalculate: true);
                    loss_expr.Backward();
                    // Update
                    trainer.Update();
                }
                loss /= 4;
                Console.WriteLine("E = " + loss);
            }// next iteration

            // Print the four options
            x.SetValue(new[] { 1f, -1f });
            Console.WriteLine("[ 1,-1] = " + y_pred.ScalarValue(fRecalculate: true));
            x.SetValue(new[] { -1f, 1f });
            Console.WriteLine("[-1, 1] = " + y_pred.ScalarValue(fRecalculate: true));
            x.SetValue(new[] { 1f, 1f });
            Console.WriteLine("[ 1, 1] = " + y_pred.ScalarValue(fRecalculate: true));
            x.SetValue(new[] { -1f, -1f });
            Console.WriteLine("[-1,-1] = " + y_pred.ScalarValue(fRecalculate: true));

            // Output the model & parameter objects to a file
            m.Save("xor.model");
        }
    }
}
