using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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

            // Expects all the parameters to be in the format -X=Y
            Dictionary<string, string> argParams = new Dictionary<string, string>();
            foreach (string arg in args) {
                string[] parts = arg.TrimStart('-').Split('=');
                argParams[parts[0]] = parts[1];
            }

            
            argParams.Add("train_file", @"D:\Temp\BigModern-raw\AllAgnonTexts-Combined-WithHeaders.txt");
            argParams.Add("test_file", @"D:\Temp\BigModern-raw\AllAgnonTexts-Combined-WithHeaders.txt");
            argParams.Add("dev_file", @"D:\Temp\BigModern-raw\AllAgnonTexts-Combined-WithHeaders.txt");

            int LAYERS = 2;
            int INPUT_DIM = 50;
            int HIDDEN_DIM = 100;
            float DROPOUT = 0.0f;

            Dictionary<string, int> d = new Dictionary<string, int>();
            d.Add("<UNK>", 0); d.Add("<s>", 1); d.Add("</s>", 2);

            // Data:
            List<List<int>> trainData = null;
            List<List<int>> devData = null;
            List<List<int>> testData = null;

            // Read the data
            if (argParams.ContainsKey("train_file")) {
                string filename = argParams["train_file"];
                Console.WriteLine("Reading training data from " + filename);
                trainData = ReadData(filename, d);
                Console.WriteLine(trainData.Count + " lines, " + trainData.Sum(l => l.Count) + " tokens, " + d.Count + " types");

                // Assuming dev data
                filename = argParams["dev_file"];
                Console.WriteLine("Reading dev data from " + filename);
                devData = ReadData(filename, d, true);
                Console.WriteLine(devData.Count + " lines, " + devData.Sum(l => l.Count) + " tokens, " + d.Count + " types");
            }

            // Test data?
            if (argParams.ContainsKey("test_file")) {
                string filename = argParams["test_file"];
                Console.WriteLine("Reading test data from " + filename);
                testData = ReadData(filename, d, fTest: true);
                Console.WriteLine(testData.Count + " lines, " + testData.Sum(l => l.Count) + " tokens, " + d.Count + " types");
            }

            // Build the model
            ParameterCollection model = new ParameterCollection();
            Trainer trainer = new SimpleSGDTrainer(model);
            // Create the language model
            LSTMBuilder lstm = new LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model);
            RNNLanguageModel lm = new RNNLanguageModel(lstm, model, d, INPUT_DIM, HIDDEN_DIM, DROPOUT);

            // Load a model?
            if (argParams.ContainsKey("model_file")) {
                string fname = argParams["model_file"];
                Console.WriteLine("Reading parameters from " + fname + "...");
                model.Load(fname);
            }

            // Train?
            if (trainData != null) {
                string modelFname = "lm_" + DROPOUT + "_" + LAYERS + "_" + INPUT_DIM + "_" + HIDDEN_DIM + ".params";
                Console.WriteLine("Parameters will be written to: " + modelFname);

                double best = double.MaxValue;
                int reportEveryI = 1;// Math.Min(100, trainData.Count);
                int devEveryIReports = 25;

                Random r = new Random();
                int reports = 0;
                for (int iEpoch = 0; iEpoch < 100; iEpoch++) { 
                    Stopwatch sw = Stopwatch.StartNew();
                    // Shuffle the train data
                    trainData = trainData.OrderBy(_ => r.Next()).ToList();

                    // New iteration
                    double loss = 0;
                    int itemsSeen = 0;
                    int charsSeen = 0;
                    // Go through entire train data
                    foreach (List<int> l in trainData) {
                        // Build the LM graph
                        Expression loss_expr = lm.BuildLMGraph(l, DROPOUT > 0f);
                        loss += loss_expr.ScalarValue();
                        charsSeen += l.Count;
                        // Backward & update
                        loss_expr.Backward();
                        trainer.Update();
                        // Report?
                        if (++itemsSeen % reportEveryI == 0) {
                            reports++;
                            Console.WriteLine("#" + reports + " [epoch=" + (iEpoch + ((double)itemsSeen / trainData.Count)) + " lr=" + trainer.LearningRate + "] E = " + (loss / charsSeen) + " ppl=" + Math.Exp(loss / charsSeen) + " ");

                            // Run dev?
                            if (reports % devEveryIReports == 0) {
                                double dloss = 0;
                                int dchars = 0;
                                foreach (var dl in devData) {
                                    loss_expr = lm.BuildLMGraph(dl, false);
                                    dloss += loss_expr.ScalarValue();
                                    dchars += dl.Count;
                                }// next dev line
                                // New best?
                                if (dloss < best) {
                                    model.Save(modelFname);
                                    best = dloss;
                                }
                                Console.WriteLine("\n***DEV [epoch=" + (iEpoch + ((double)itemsSeen / trainData.Count)) + "] E = " + (dloss / dchars) + " ppl=" + Math.Exp(dloss / dchars) + " ");
                            }// end of dev
                        }//end of report
                    }// next l
                }// next epoch
            }// end of train
            // Test?
            if (testData != null) {
                Console.WriteLine("Evaluating test data...");
                double tloss = 0;
                double tchars = 0;
                foreach (var l in testData) {
                    Expression loss_expr = lm.BuildLMGraph(l, false);
                    tloss += loss_expr.ScalarValue();
                    tchars += l.Count;
                }// next test item

                Console.WriteLine("TEST                -LLH = " + tloss);
                Console.WriteLine("TEST CROSS ENTOPY (NATS) = " + (tloss / tchars));
                Console.WriteLine("TEST                 PPL = " + Math.Exp(tloss / tchars));
            }// end of test
        }

        private static List<List<int>> ReadData(string filename, Dictionary<string, int> d, bool fTest = false) {
            List<List<int>> data = new List<List<int>>();
            foreach (string line in File.ReadLines(filename)) {
                string[] words = line.Split(' ');
                // Convert to ints
                List<int> curLine = new List<int>();
                foreach (string w in words) {
                    if (!fTest && !d.ContainsKey(w))
                        d.Add(w, d.Count);
                    curLine.Add(d.ContainsKey(w) ? d[w] : d["<UNK>"]);
                }// next word
                // Add it in
                data.Add(curLine);
            }// next line
            return data;
        }

        class RNNLanguageModel {
            private RNNBuilder builder;
            private LookupParameter lp;
            private Parameter p_R;
            private Parameter p_bias;
            private float dropout;
            private Dictionary<string, int> d;
            private Dictionary<int, string> di2W;
            public RNNLanguageModel(RNNBuilder builder, ParameterCollection model, Dictionary<string, int> vocab, int INPUT_DIM, int HIDDEN_DIM, float dropout) {
                this.builder = builder;
                this.dropout = dropout;
                this.d = vocab;
                this.di2W = d.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
                lp = model.AddLookupParameters(vocab.Count, new[] { INPUT_DIM });
                p_R = model.AddParameters(new[] { vocab.Count, HIDDEN_DIM });
                p_bias = model.AddParameters(new[] { vocab.Count });
            }

            // return Expression of total loss
            public Expression BuildLMGraph(List<int> sent, bool fApplyDropout) {
                // Renew the computation graph
                dy.RenewCG();

                // hidden -> word rep parameter
                Expression R = dy.parameter(p_R); 
                // word bias
                Expression bias = dy.parameter(p_bias);

                // Build the collection of losses
                List<Expression> errs = new List<Expression>();

                // Start the initial state with the a <s> tag
                RNNState state = builder.GetInitialState().AddInput(lp[d["<s>"]]);
                // Go through all the inputs
                for (int t = 0; t < sent.Count; t++) {
                    // Regular softmax
                    Expression u_t = dy.affine_transform(bias, R, state.Output());
                    errs.Add(dy.pickneglogsoftmax(u_t, sent[t]));
                    // Add the next item in
                    state = state.AddInput(dy.lookup(lp, sent[t]));
                }// next t
                // Add the last </s> tag
                Expression u_last = dy.affine_transform(bias, R, state.Output());
                errs.Add(dy.pickneglogsoftmax(u_last, d["</s>"]));

                // Run the sum
                return dy.esum(errs);
            }
            public void RandomSample(int maxLen = 200) {
                // Renew the computation graph
                dy.RenewCG();

                // hidden -> word rep parameter
                Expression R = dy.parameter(p_R);
                // word bias
                Expression bias = dy.parameter(p_bias);

                Random r = new Random();
                // Start with an <s>
                RNNState state = builder.GetInitialState().AddInput(lp[d["<s>"]]);
                int cur = d["<s>"], len = 0;
                while (len < maxLen) {
                    // Regular softmax
                    Expression u_t = dy.affine_transform(bias, R, state.Output());
                    Expression dist_expr = dy.softmax(u_t);
                    float[] dist = dist_expr.VectorValue();
                    // Get a random between 0->1, sample the next item
                    double p = r.NextDouble();
                    for (cur = 0; cur < dist.Length; cur++) {
                        p -= dist[cur];
                        if (p < 0) break;
                    }
                    if (cur == dist.Length) cur = d["</s>"];
                    // Are we at the end?
                    if (cur == d["</s>"])
                        break;
                    len++;
                    // Output the chracter
                    Console.Write((len == 1 ? "" : " ") + di2W[cur]);
                }// next prediction
                Console.WriteLine();
            }
        }
    }
}
