# Requirements

from __future__ import unicode_literals, print_function, division
import io
import unicodedata
import re
import random
import dynet as dy
import time
import math
r = random.SystemRandom()

# Data Preparation

SOS_token = 0
EOS_token = 1


class Lang(object):

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):

    print("Reading lines...")
    lines = io.open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


MAX_LENGTH = 10
eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filterPair(p):

    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):

    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(r.choice(pairs))

# Model


class EncoderRNN(object):

    def __init__(self, in_vocab, hidden_dim, model):
        self.in_vocab = in_vocab
        self.hidden_dim = hidden_dim
        self.embedding_enc = model.add_lookup_parameters((self.in_vocab,
                                                          self.hidden_dim))
        self.rnn_enc = dy.GRUBuilder(1, self.hidden_dim, self.hidden_dim,
                                     model)

    def __call__(self, inputs, hidden):
        input_embed = dy.lookup(self.embedding_enc, inputs)
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
        self.embedding_dec = model.add_lookup_parameters((self.out_vocab,
                                                          self.hidden_dim))
        self.w_attn = model.add_parameters((self.max_length,
                                            self.hidden_dim * 2))
        self.b_attn = model.add_parameters((self.max_length,))
        self.w_attn_combine = model.add_parameters((self.hidden_dim,
                                                    self.hidden_dim * 2))
        self.b_attn_combine = model.add_parameters((self.hidden_dim,))
        self.rnn_dec = dy.GRUBuilder(1, self.hidden_dim, self.hidden_dim,
                                     model)
        self.w_dec = model.add_parameters((self.out_vocab, self.hidden_dim))
        self.b_dec = model.add_parameters((self.out_vocab,))

    def __call__(self, inputs, hidden, encoder_outptus, dropout=False):
        input_embed = dy.lookup(self.embedding_dec, inputs)
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

        return output, state_dec.h(), attn_weights

    def initHidden(self):
        return [dy.zeros(self.hidden_dim)]


def indexesFromSentence(lang, sentence):

    return [lang.word2index[word] for word in sentence.split(" ")] + \
           [EOS_token]


def indexesFromPair(pair):

    input_indexes = indexesFromSentence(input_lang, pair[0])
    target_indexes = indexesFromSentence(output_lang, pair[1])
    return (input_indexes, target_indexes)

# Training the Model


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

    if r.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs, dropout=True)
            losses.append(-dy.log(dy.pick(decoder_output, targets[i])))
            decoder_input = targets[i]
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs, dropout=True)
            losses.append(-dy.log(dy.pick(decoder_output, targets[i])))
            probs = decoder_output.vec_value()
            decoder_input = probs.index(max(probs))
            if decoder_input == EOS_token:
                break

    loss = dy.esum(losses)/len(losses)
    loss.backward()
    trainer.update()

    return loss.value()

# Helper Function to Print Time


def asMinutes(s):
    m = math.floor(s/60)
    s -= m*60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))

# Whole Training Process


def trainIters(encoder, decoder, trainer, n_iters, print_every=1000,
               plot_every=100):

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    training_pairs = [indexesFromPair(r.choice(pairs))
                      for _ in range(n_iters)]

    for iteration in range(1, n_iters+1):

        training_pair = training_pairs[iteration-1]
        inputs = training_pair[0]
        targets = training_pair[1]

        loss = train(inputs, targets, encoder, decoder, trainer)

        print_loss_total += loss
        plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print("%s (%d %d%%) %.4f" % (timeSince(start, iteration/n_iters),
                                         iteration, iteration/n_iters*100,
                                         print_loss_avg))

        if iteration % plot_every == 0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

# Evaluation


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    dy.renew_cg()

    encoder_hidden = encoder.initHidden()

    inputs = indexesFromSentence(input_lang, sentence)
    input_length = len(inputs)

    encoder_outputs = [dy.zeros(hidden_dim) for _ in range(max_length)]

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(inputs[i], encoder_hidden)
        encoder_outputs[i] = encoder_output

    encoder_outputs = dy.concatenate(encoder_outputs, 1)

    decoder_input = SOS_token
    decoder_hidden = encoder_hidden

    decoder_words = []
    decoder_attentions = [dy.zeros(max_length) for _ in range(max_length)]

    for i in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, dropout=False)
        decoder_attentions[i] = decoder_attention
        probs = decoder_output.vec_value()
        pred = probs.index(max(probs))
        if pred == EOS_token:
            decoder_words.append("<EOS>")
            break
        else:
            decoder_words.append(output_lang.index2word[pred])
        decoder_input = pred

    return decoder_words


def evaluationRandomly(encoder, decoder, n=10):

    for _ in range(n):
        pair = r.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")

# Start Training and Evaluating


model = dy.ParameterCollection()
hidden_dim = 256
encoder = EncoderRNN(input_lang.n_words, hidden_dim, model)
decoder = AttnDecoderRNN(hidden_dim, output_lang.n_words, model)
trainer = dy.SimpleSGDTrainer(model, learning_rate=0.2)

trainIters(encoder, decoder, trainer, 100000, print_every=5000)

evaluationRandomly(encoder, decoder)
