# Translation with a Sequence to Sequence Network and Attention

Here is a Dynet version of the [PyTorch tutorial example "Translation with a Sequence to Sequence Network and Attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

## Data Preparation 

The data for this project is a set of many thousands of English to French translation pairs. Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory. This part is exactly the same as that in PyTorch tutorial.

## The Seq2Seq Model

### The Encoder

<pre>
class EncoderRNN(object):

    def __init__(self, in_vocab, hidden_dim, model):
        self.embedding_enc = model.add_lookup_parameters((in_vocab, hidden_dim))
        self.rnn_enc = dy.GRUBuilder(1, hidden_dim, hidden_dim, model)

    def __call__(self, input, hidden):
        input_embed = dy.lookup(self.embedding_enc, input)
        state_enc = self.rnn_enc.initial_state(vecs=hidden)
        state_enc = state_enc.add_input(input_embed)
        return state_enc.output(), state_enc.h()

    def initHidden(self):
        return [dy.zeros(hidden_dim)]
</pre>

### The Decoder (without attention mechanism)

<pre>
class DecoderRNN(object):

    def __init__(self, hidden_dim, out_vocab, model):
        self.embedding_dec = model.add_lookup_parameters((out_vocab, hidden_dim))
        self.rnn_dec = dy.GRUBuilder(1, hidden_dim, hidden_dim, model)
        self.w_dec = model.add_parameters((out_vocab, hidden_dim))
        self.b_dec = model.add_parameters((out_vocab,))

    def __call__(self, input, hidden):
        input_embed = dy.lookup(self.embedding_dec, input)
        input_embed = dy.rectify(input_embed)
        state_dec = self.rnn_dec.initial_state(vecs=hidden)
        state_dec = state_dec.add_input(input_embed)
        w_dec = dy.parameter(self.w_dec)
        b_dec = dy.parameter(self.b_dec)
        output = state_dec.output()
        # output = dy.log_softmax(w_dec*output)
        output = dy.softmax(w_dec * output + b_dec)
        return output, state_dec.h()

    def initHidden(self):
        return [dy.zeros(hidden_dim)]
</pre>
