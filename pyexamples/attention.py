import pycnn as pc
import random

EOS = "<EOS>"
characters = list("abcdefghijklmnopqrstuvwxyz ")
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

LSTM_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 16
STATE_SIZE = 32

model = pc.Model()

model.add_lookup_parameters("lookup", (VOCAB_SIZE, EMBEDDINGS_SIZE))

enc_fwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

enc_bwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

model.add_parameters("attention_w", (1, STATE_SIZE*2+STATE_SIZE*LSTM_NUM_OF_LAYERS*2))

dec_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2, STATE_SIZE, model)

model.add_parameters("decoder_w", (VOCAB_SIZE, STATE_SIZE))
model.add_parameters("decoder_b", (VOCAB_SIZE))


def embedd_sentence(model, sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]

    pc.renew_cg()

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


def attend(model, vectors, state):
    w = pc.parameter(model['attention_w'])
    attention_weights = []
    for vector in vectors:
        attention_input = pc.concatenate([vector, pc.concatenate(list(state.s()))])
        attention_weight = pc.tanh(w * attention_input)
        attention_weights.append(attention_weight)
    attention_weights = pc.concatenate(attention_weights)
    vectors = pc.softmax(pc.esum([vector*attention_weight for vector, attention_weight in zip(vectors, attention_weights)]))
    return vectors


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


def train(model, sentence):
    trainer = pc.SimpleSGDTrainer(model)
    for i in xrange(400):
        embedded = embedd_sentence(model, sentence)
        encoded = encode_sentence(model, enc_fwd_lstm, enc_bwd_lstm, embedded)
        loss = decode(model, dec_lstm, encoded, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print loss_value
            print generate(model, sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)


train(model, "it is working")


