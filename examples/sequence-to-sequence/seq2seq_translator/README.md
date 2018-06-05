# Seq2seq Translator Benchmarks

Here is the comparison between Dynet and PyTorch on the [seq2seq translator example](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

The data we used is a set of many thousands of English to French translation pairs. Download the data from [here](https://download.pytorch.org/tutorial/data.zip) and extract it to the current directory.

## Usage (Dynet)

The architecture of the dynet model `seq2seq_dynet.py` is the same as that in [PyTorch Example](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). We here implement the attention mechanism in the model.

The architecture of the dynet model is shown as follows. 

```python
class EncoderRNN(object):

    def __init__(self, in_vocab, hidden_dim, model):
        self.in_vocab = in_vocab
        self.hidden_dim = hidden_dim
        self.embedding_enc = model.add_lookup_parameters((self.in_vocab, self.hidden_dim))
        self.rnn_enc = dy.GRUBuilder(1, self.hidden_dim, self.hidden_dim, model)

    def __call__(self, input, hidden):
        input_embed = dy.lookup(self.embedding_enc, input)
        state_enc = self.rnn_enc.initial_state(vecs=hidden)
        state_enc = state_enc.add_input(input_embed)
        return state_enc.output(), state_enc.h()

    def initHidden(self):
        return [dy.zeros(self.hidden_dim)] 
   
DROPOUT_RATE = 0.1

class AttnDecoderRNN(object):

    def __init__(self, hidden_dim, out_vocab, model, max_length=MAX_LENGTH):
        self.hidden_dim = hidden_dim
        self.out_vocab = out_vocab
        self.max_length = max_length
        self.embedding_dec = model.add_lookup_parameters((self.out_vocab, self.hidden_dim))
        self.w_attn = model.add_parameters((self.max_length, self.hidden_dim * 2))
        self.b_attn = model.add_parameters((self.max_length,))
        self.w_attn_combine = model.add_parameters((self.hidden_dim, self.hidden_dim * 2))
        self.b_attn_combine = model.add_parameters((self.hidden_dim,))
        self.rnn_dec = dy.GRUBuilder(1, self.hidden_dim, self.hidden_dim, model)
        self.w_dec = model.add_parameters((self.out_vocab, self.hidden_dim))
        self.b_dec = model.add_parameters((self.out_vocab,))

    def __call__(self, input, hidden, encoder_outptus, dropout=False):
        input_embed = dy.lookup(self.embedding_dec, input)
        if dropout:
            input_embed = dy.dropout(input_embed, DROPOUT_RATE)
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
        return [dy.zeros(self.hidden_dim)]
```

Install the GPU version of Dynet according to the instructions on the [official website](http://dynet.readthedocs.io/en/latest/python.html#installing-a-cutting-edge-and-or-gpu-version).

Then, run the training:

<pre>
python seq2seq_dynet.py --dynet_gpus 1
</pre>

## Usage (PyTorch)

The code of `seq2seq_pytorch.py` follows the same line in [PyTorch Example](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). 

The architecture of the pytorch model is shown as follows.

```python
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

Install CUDA version of PyTorch according to the instructions on the [official website](http://pytorch.org/).

Then, run the training:

<pre>
python seq2seq_pytorch.py
</pre>

## Performance

We run our codes on a desktop with NVIDIA TITAN X. We here have D stands for Dynet and P stands for PyTorch.

| Time (D) | Iteration (D) | Loss (D) | Time (P) | Iteration (P) | Loss (P)|
| --- | --- | --- | --- | --- | --- |
| 0m 26s | 5000 5% | 3.3565 | 1m 30s | 5000 5% | 2.8794 |
| 0m 53s | 10000 10% | 2.7376 | 2m 55s | 10000 10% | 2.3103 |
| 1m 21s | 15000 15% | 2.4912 | 4m 5s | 15000 15% | 1.9939 |
| 1m 48s | 20000 20% | 2.2536 | 5m 16s | 20000 20% | 1.7537 |
| 2m 16s | 25000 25% | 2.0537 | 6m 27s | 25000 25% | 1.5796 |
| 2m 44s | 30000 30% | 1.8832 | 7m 39s | 30000 30% | 1.3795 |
| 3m 12s | 35000 35% | 1.7232 | 9m 13s | 35000 35% | 1.2712 |
| 3m 40s | 40000 40% | 1.5833 | 10m 31s | 40000 40% | 1.1374 |
| 4m 8s | 45000 45% | 1.4360 | 11m 41s | 45000 45% | 1.0215 |
| 4m 36s | 50000 50% | 1.2916 | 12m 53s | 50000 50% | 0.9307 |
| 5m 4s | 55000 55% | 1.2023 | 14m 5s | 55000 55% | 0.8312 |
| 5m 33s | 60000 60% | 1.1186 | 15m 17s | 60000 60% | 0.7879 |
| 6m 1s | 65000 65% | 1.0435 | 16m 48s | 65000 65% | 0.7188 |
| 6m 30s | 70000 70% | 0.9348 | 18m 6s | 70000 70% | 0.6532 |
| 6m 58s | 75000 75% | 0.8634 | 19m 18s | 75000 75% | 0.6273 |
| 7m 26s | 80000 80% | 0.8323 | 20m 34s | 80000 80% | 0.6021 |
| 7m 55s | 85000 85% | 0.7610 | 21m 44s | 85000 85% | 0.5210 |
| 8m 23s | 90000 90% | 0.7377 | 22m 55s | 90000 90% | 0.5054 |
| 8m 52s | 95000 95% | 0.6666 | 24m 9s | 95000 95% | 0.4417 |
| 9m 21s | 100000 100% | 0.6237 | 25m 24s | 100000 100% | 0.4297 |

We then show some evaluation results as follows.

Format:

<pre>
> input 
= target 
< output
</pre>

### Dynet

```
> elle est infirmiere .
= she is a nurse .
< she is a nurse . <EOS>

> tu n es pas contrariee si ?
= you re not upset are you ?
< you re not upset are you re not upset are

> j en ai termine avec mon travail .
= i am through with my work .
< i am through with my work . <EOS>

> je ne l invente pas .
= i m not making that up .
< i m not making up . <EOS>

> elles ont peur de moi .
= they re afraid of me .
< they re afraid of me . <EOS>

> on va jouer au tennis .
= we re going to play tennis .
< we are going tennis . <EOS>

> j ai une assuetude .
= i m addicted .
< i m addicted . <EOS>

> elles sont en train de vous chercher .
= they re looking for you .
< they re looking for you . <EOS>

> elle semble riche .
= she seems rich .
< she seems rich . <EOS>

> vous etes bizarre .
= you re weird .
< you re weird . <EOS>
```

### PyTorch

```
> il est deja marie .
= he s already married .
< he s already married . <EOS>

> on le dit decede .
= he is said to have died .
< he are said to have died . <EOS>

> il est trop saoul .
= he s too drunk .
< he s too drunk . <EOS>

> je suis assez heureux .
= i m happy enough .
< i m happy happy . <EOS>

> je n y suis pas interessee .
= i m not interested in that .
< i m not interested in that . <EOS>

> il a huit ans .
= he s eight years old .
< he is thirty . <EOS>

> je ne suis pas differente .
= i m no different .
< i m no different . <EOS>

> je suis heureux que vous l ayez aime .
= i m happy you liked it .
< i m happy you liked it . <EOS>

> ils peuvent chanter .
= they re able to sing .
< they re able to sing . <EOS>

> vous etes tellement belle dans cette robe !
= you re so beautiful in that dress .
< you re so beautiful in that dress . <EOS>
```
