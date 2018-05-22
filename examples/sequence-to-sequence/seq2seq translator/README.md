# Translation with a Sequence to Sequence Network and Attention

Here is a Dynet version of the [PyTorch tutorial example "Translation with a Sequence to Sequence Network and Attention"](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

## Data Preparation 

The data for this project is a set of many thousands of English to French translation pairs. Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory. This part is exactly the same as that in PyTorch tutorial.

## The Seq2Seq Model

A Sequence to Sequence network, or seq2seq network, or Encoder Decoder network, is a model consisting of two RNNs called the encoder and decoder. The encoder reads an input sequence and outputs a single vector, and the decoder reads that vector to produce an output sequence.

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

### The Decoder (with attention mechanism)

<pre>
class AttnDecoderRNN(object):

    def __init__(self, hidden_dim, out_vocab, model, dropout_p=0.1, max_length=MAX_LENGTH):
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding_dec = model.add_lookup_parameters((out_vocab, hidden_dim))
        self.w_attn = model.add_parameters((self.max_length, hidden_dim * 2))
        self.b_attn = model.add_parameters((self.max_length,))
        self.w_attn_combine = model.add_parameters((hidden_dim, hidden_dim * 2))
        self.b_attn_combine = model.add_parameters((hidden_dim,))
        self.rnn_dec = dy.GRUBuilder(1, hidden_dim, hidden_dim, model)
        self.w_dec = model.add_parameters((out_vocab, hidden_dim))
        self.b_dec = model.add_parameters((out_vocab,))

    def __call__(self, input, hidden, encoder_outptus):
        input_embed = dy.lookup(self.embedding_dec, input)
        input_embed = dy.dropout(input_embed, self.dropout_p)
        input_cat = dy.concatenate([input_embed, hidden[0]])
        w_attn = dy.parameter(self.w_attn)
        b_attn = dy.parameter(self.b_attn)
        attn_weights = dy.softmax(w_attn * input_cat + b_attn)
        attn_applied = encoder_outptus * attn_weights
        output = dy.concatenate([input_embed, attn_applied])
        w_attn_combine = dy.parameter(self.w_attn_combine)
        b_attn_combine = dy.parameter(self.b_attn_combine)
        output = w_attn_combine * output + b_attn_combine
        output = dy.rectify(output)
        state_dec = self.rnn_dec.initial_state(vecs=hidden)
        state_dec = state_dec.add_input(output)
        w_dec = dy.parameter(self.w_dec)
        b_dec = dy.parameter(self.b_dec)
        output = state_dec.output()
        output = dy.softmax(w_dec * output + b_dec)

        return output, hidden, attn_weights

    def initHidden(self):
        return [dy.zeros(hidden_dim)]
</pre>

## Training

### Preparing Training Data

<pre>
def indexesFromSentence(lang, sentence):

    return [lang.word2index[word] for word in sentence.split(" ")]

def indexesFromPair(pair):

    input_indexes = indexesFromSentence(input_lang, pair[0])
    target_indexes = indexesFromSentence(output_lang, pair[1])
    return (input_indexes, target_indexes)
</pre>

### Training the Model (without attention mechanism)

<pre>
teacher_forcing_ratio = 0.5

def train(inputs, targets, encoder, decoder, trainer):

    dy.renew_cg()

    encoder_hidden = encoder.initHidden()

    input_length = len(inputs)
    target_length = len(targets)

    encoder_outputs = []

    losses = []

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(inputs[i], encoder_hidden)
        encoder_outputs.append(encoder_output)

    decoder_input = SOS_token
    decoder_hidden = encoder_hidden

    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            losses.append(-dy.log(dy.pick(decoder_output, targets[i])))
            decoder_input = targets[i]
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            losses.append(-dy.log(dy.pick(decoder_output, targets[i])))
            probs = decoder_output.vec_value()
            decoder_input = probs.index(max(probs))
            if decoder_input == EOS_token:
                break

    loss = dy.esum(losses)/len(losses)
    loss.backward()
    trainer.update()

    return loss.value()
</pre>

### Training the Model (with attention mechanism)

<pre>
teacher_forcing_ratio = 0.5

def train(inputs, targets, encoder, decoder, trainer, max_length=MAX_LENGTH):

    dy.renew_cg()

    encoder_hidden = encoder.initHidden()

    input_length = len(inputs)
    target_length = len(targets)

    encoder_outputs = [dy.zeros(hidden_dim) for _ in range(max_length)]

    losses = []

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(inputs[i], encoder_hidden)
        encoder_outputs[i] = encoder_output

    encoder_outputs = dy.concatenate(encoder_outputs, 1)

    decoder_input = SOS_token
    decoder_hidden = encoder_hidden

    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            losses.append(-dy.log(dy.pick(decoder_output, targets[i])))
            decoder_input = targets[i]
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            losses.append(-dy.log(dy.pick(decoder_output, targets[i])))
            probs = decoder_output.vec_value()
            decoder_input = probs.index(max(probs))
            if decoder_input == EOS_token:
                break

    loss = dy.esum(losses)/len(losses)
    loss.backward()
    trainer.update()

    return loss.value()
</pre>
