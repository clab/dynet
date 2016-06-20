import pycnn as pc
import random

EOS = "<EOS>"
characters = list("abcdefghijklmnopqrstuvwxyz ")
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

LSTM_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 32
STATE_SIZE = 32
ATTENTION_SIZE = 32

model = pc.Model()

enc_fwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
enc_bwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

dec_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2, STATE_SIZE, model)

model.add_lookup_parameters("lookup", (VOCAB_SIZE, EMBEDDINGS_SIZE))
model.add_parameters("attention_w1", (ATTENTION_SIZE, STATE_SIZE*2))
model.add_parameters("attention_w2", (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
model.add_parameters("attention_v", (1, ATTENTION_SIZE))
model.add_parameters("decoder_w", (VOCAB_SIZE, STATE_SIZE))
model.add_parameters("decoder_b", (VOCAB_SIZE))


def embedd_sentence(model, sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]

    lookup = model["lookup"]

    return [lookup[char] for char in sentence]


def run_lstm(model, init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(model, enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = [sentence[i] for i in range(len(sentence)-1, -1, -1)]

    fwd_vectors = run_lstm(model, enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(model, enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = [bwd_vectors[i] for i in range(len(bwd_vectors)-1, -1, -1)]
    vectors = [pc.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(model, input_vectors, state):
    w1 = pc.parameter(model['attention_w1'])
    w2 = pc.parameter(model['attention_w2'])
    v = pc.parameter(model['attention_v'])
    attention_weights = []

    w2dt = w2*pc.concatenate(list(state.s()))
    for input_vector in input_vectors:
        attention_weight = v*pc.tanh(w1*input_vector + w2dt)
        attention_weights.append(attention_weight)
    attention_weights = pc.softmax(pc.concatenate(attention_weights))
    output_vectors = pc.esum([vector*attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
    return output_vectors


def decode(model, dec_lstm, vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = pc.parameter(model["decoder_w"])
    b = pc.parameter(model["decoder_b"])

    s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE*2))

    loss = []
    for char in output:
        vector = attend(model, vectors, s)

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = pc.softmax(out_vector)
        loss.append(-pc.log(pc.pick(probs, char)))
    loss = pc.esum(loss)
    return loss


def generate(model, input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    def sample(probs):
        rnd = random.random()
        for i, p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    embedded = embedd_sentence(model, input)
    encoded = encode_sentence(model, enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = pc.parameter(model["decoder_w"])
    b = pc.parameter(model["decoder_b"])

    s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE * 2))
    out = ''
    count_EOS = 0
    for i in range(len(input)*2):
        if count_EOS == 2: break
        vector = attend(model, encoded, s)

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = pc.softmax(out_vector)
        probs = probs.vec_value()
        next_char = sample(probs)
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def get_loss(model, input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    pc.renew_cg()
    embedded = embedd_sentence(model, input_sentence)
    encoded = encode_sentence(model, enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(model, dec_lstm, encoded, output_sentence)


def train(model, sentence):
    trainer = pc.SimpleSGDTrainer(model)
    for i in xrange(600):
        loss = get_loss(model, sentence, sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print loss_value
            print generate(model, sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)


train(model, "it is working")


