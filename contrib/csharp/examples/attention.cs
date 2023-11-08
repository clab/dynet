using System;
using System.Collections.Generic;
using System.Linq;
using dynetsharp;
using dy = dynetsharp.DynetFunctions;

namespace DynetSharpExamples {
    class Program {
        static int VOCAB_SIZE, LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, ATTENTION_SIZE;
        static void Main(string[] args) {
            DynetParams.FromArgs(args).Initialize();
            // Alternatively, can initialize and it directly, e.g:
            // DynetParams dp = new DynetParams();
            // dp.AutoBatch = true;
            // dp.MemDescriptor = "768";
            // dp.Initialize();

            const string EOS = "<EOS>";
            List<string> characters = "abcdefghijklmnopqrstuvwxyz ".Select(c => c.ToString()).ToList();
            characters.Add(EOS);

            // Lookup - dictionary
            Dictionary<string, int> c2i = Enumerable.Range(0, characters.Count).ToDictionary(i => characters[i], i => i);

            // Define the variables
            VOCAB_SIZE = characters.Count;
            LSTM_NUM_OF_LAYERS = 2;
            EMBEDDINGS_SIZE = 32;
            STATE_SIZE = 32;
            ATTENTION_SIZE = 32;

            // ParameterCollection (all the model parameters).
            ParameterCollection m = new ParameterCollection();
            // A class defined locally used to contain all the parameters to transfer
            // them between functions and avoid global variables
            ParameterGroup pg = new ParameterGroup();
            pg.c2i = c2i;
            pg.i2c = characters;
            pg.EOS = EOS;

            // LSTMs
            pg.enc_fwd_lstm = new LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, m);
            pg.enc_bwd_lstm = new LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, m);

            pg.dec_lstm = new LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + EMBEDDINGS_SIZE, STATE_SIZE, m );

            // Create the parameters
            pg.input_lookup = m.AddLookupParameters(VOCAB_SIZE, new[] { EMBEDDINGS_SIZE });
            pg.attention_w1 = m.AddParameters(new[] { ATTENTION_SIZE, STATE_SIZE * 2 });
            pg.attention_w2 = m.AddParameters(new[] { ATTENTION_SIZE, STATE_SIZE * 2 * LSTM_NUM_OF_LAYERS });
            pg.attention_v = m.AddParameters(new[] { 1, ATTENTION_SIZE });
            pg.decoder_W = m.AddParameters(new[] { VOCAB_SIZE, STATE_SIZE });
            pg.decoder_b = m.AddParameters(new[] { VOCAB_SIZE });
            pg. output_lookup = m.AddLookupParameters(VOCAB_SIZE, new[] { EMBEDDINGS_SIZE });

            Trainer trainer = new SimpleSGDTrainer(m);

            // For good practice, renew the computation graph
            dy.RenewCG();

            // Train
            string trainSentence = "it is working";
            // Run 600 epochs
            for (int iEpoch = 0; iEpoch < 600; iEpoch++) {
                // Loss
                Expression loss = CalculateLoss(trainSentence, trainSentence, pg);
                // Forward, backward, update
                float lossValue = loss.ScalarValue();
                loss.Backward();
                trainer.Update();
                if (iEpoch % 20 == 0) {
                    Console.WriteLine(lossValue);
                    Console.WriteLine(GenerateSentence(trainSentence, pg)); 
                }
            }// next epoch

        }

        private static Expression CalculateLoss(string inputSentence, string outputSentence, ParameterGroup pg) {
            dy.RenewCG();
            // Embed, encode, and decode
            List<Expression> embeds = EmbedSentence(inputSentence, pg);
            List<Expression> encodings = EncodeSentence(embeds, pg);
            return DecodeSentence(encodings, outputSentence, pg);
        }
        
        private static List<Expression> EmbedSentence(string inputSentence, ParameterGroup pg) {
            // Convert to embeddings
            List<Expression> embeds = inputSentence.Select(c => pg.input_lookup[pg.c2i[c.ToString()]]).ToList();
            // Pad with eos
            embeds.Insert(0, pg.input_lookup[pg.c2i[pg.EOS]]);
            embeds.Add(pg.input_lookup[pg.c2i[pg.EOS]]);

            return embeds;
        }
        private static List<Expression> EncodeSentence(List<Expression> embeds, ParameterGroup pg) {
            // Go forwards
            List<Expression> fwdOutputs = pg.enc_fwd_lstm.GetInitialState().Transduce(embeds);
            // Go bwds, then reverse
            List<Expression> bwdOutputs = pg.enc_bwd_lstm.GetInitialState().Transduce(embeds.Reverse<Expression>().ToArray());
            bwdOutputs.Reverse();

            // Concatenate them
            return fwdOutputs.Zip(bwdOutputs, (fwd,bwd) => dy.concatenate(fwd, bwd)).ToList();
        }

        private static Expression DecodeSentence(List<Expression> encodings, string outputSentence, ParameterGroup pg) {
            // Pad the output *only at end* with eos
            List<string> output = outputSentence.Select(c => c.ToString()).ToList();
            output.Add(pg.EOS);

            // Create the matrix - of all the context vectors
            Expression inputMat = dy.concatenate_cols(encodings);
            // Each attention is an activation layer on top of the sum of w1*inputMat and w2*state
            // Since w1*inputMat is static - calculate it here
            Expression w1dt = pg.attention_w1 * inputMat;

            // Create the initial state of the decoder
            RNNState decState = pg.dec_lstm.GetInitialState();
            // Run the EOS through (attend initial will be zeros)
            decState = decState.AddInput(dy.concatenate(dy.zeros(new[] { STATE_SIZE * 2 }), pg.output_lookup[pg.c2i[pg.EOS]]));

            List<Expression> losses = new List<Expression>();
            // Go through and decode
            Expression prev = pg.output_lookup[pg.c2i[pg.EOS]];
            foreach (string outS in output) {
                // Create the input
                Expression inputVec = dy.concatenate(Attend(inputMat, w1dt, decState, pg), prev);
                // Run through LSTM + linear layer
                decState = decState.AddInput(inputVec);
                Expression outputVec = dy.softmax(pg.decoder_W * decState.Output() + pg.decoder_b);
                // Loss & next
                losses.Add(-dy.log(dy.pick(outputVec, pg.c2i[outS])));
                prev = pg.output_lookup[pg.c2i[outS]];
            }// next output

            return dy.sum(losses);

        }
        private static string GenerateSentence(string inputSentence, ParameterGroup pg) {
            dy.RenewCG();

            List<Expression> embeds = EmbedSentence(inputSentence, pg);
            List<Expression> encodings = EncodeSentence(embeds, pg);

            // Create the matrix - of all the context vectors
            Expression inputMat = dy.concatenate_cols(encodings);
            // Each attention is an activation layer on top of the sum of w1*inputMat and w2*state
            // Since w1*inputMat is static - calculate it here
            Expression w1dt = pg.attention_w1 * inputMat;

            // Create the initial state of the decoder
            RNNState decState = pg.dec_lstm.GetInitialState();
            // Run the EOS through (attend initial will be zeros)
            decState = decState.AddInput(dy.concatenate(dy.zeros(new[] { STATE_SIZE * 2 }), pg.output_lookup[pg.c2i[pg.EOS]]));

            List<string> output = new List<string>();
            Expression prev = pg.output_lookup[pg.c2i[pg.EOS]];
            // Go through and decode
            for (int i = 0; i < inputSentence.Length * 2; i++) { 
                // Create the input
                Expression inputVec = dy.concatenate(Attend(inputMat, w1dt, decState, pg), prev);
                // Run through LSTM + linear layer
                decState = decState.AddInput(inputVec);
                Expression outputVec = dy.softmax(pg.decoder_W * decState.Output() + pg.decoder_b);
                // Get the predictions
                int max = Argmax(outputVec.VectorValue());
                if (max == pg.c2i[pg.EOS]) break;
                output.Add(pg.i2c[max]);
                prev = pg.output_lookup[max];
            }// next output

            return string.Join("", output);
        }
        static int Argmax(float[] vec) {
            return Enumerable.Range(0, vec.Length).OrderByDescending(i => vec[i]).First();
        }
        private static Expression Attend(Expression inputMat, Expression w1dt, RNNState decState, ParameterGroup pg) {
            // We have w1dt which is Attention x len(seq) 
            // Now, concate the hidden layers from the decoder and multiply that by w2 
            // will give us 1xAttention
            Expression w2dt = pg.attention_w2 * dy.concatenate(decState.GetS());
            // Add that to each column, run through an activation layer, and those are our
            // "energies" (we have to transpose in order to get the vector dimensions)
            Expression unnormalized = dy.transpose(pg.attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)));
            Expression attentionWeights = dy.softmax(unnormalized);
            // Apply the weights and return the new weighted ci
            return inputMat * attentionWeights;
        }

        class ParameterGroup {
            internal string EOS;
            internal List<string> i2c;
            internal Dictionary<string, int> c2i;
            internal LSTMBuilder enc_fwd_lstm;
            internal LSTMBuilder enc_bwd_lstm;
            internal LSTMBuilder dec_lstm;
            internal LookupParameter input_lookup;
            internal Parameter attention_w1;
            internal Parameter attention_w2;
            internal Parameter attention_v;
            internal Parameter decoder_W;
            internal Parameter decoder_b;
            internal LookupParameter output_lookup;
        }
    }
}
