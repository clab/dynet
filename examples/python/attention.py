import dynet as dy
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

model = dy.Model()

enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters( (1, ATTENTION_SIZE))
decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
decoder_b = model.add_parameters( (VOCAB_SIZE))
output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))


def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]

    global input_lookup

    return [input_lookup[char] for char in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_vectors, state):
    global attention_w1
    global attention_w2
    global attention_v
    w1 = dy.parameter(attention_w1)
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)
    attention_weights = []

    w2dt = w2*dy.concatenate(list(state.s()))
    for input_vector in input_vectors:
        attention_weight = v*dy.tanh(w1*input_vector + w2dt)
        attention_weights.append(attention_weight)
    attention_weights = dy.softmax(dy.concatenate(attention_weights))
    output_vectors = dy.esum([vector*attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
    return output_vectors


def make_decoder(input_sequence):
    global decoder_w
    global decoder_w
    global enc_fwd_lstm
    global enc_bwd_lstm
    global dec_lstm

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)

    embedded = embed_sentence(input_sequence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))
    while True:
        vector = dy.concatenate([attend(encoded, s), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        yield probs  # output decoding probabilities
        prev_char = yield  # expect feedback char through .send(char) to continue decoding
        last_output_embeddings = output_lookup[prev_char]


def train_decode(input_sequence, output_sequence):
    output_sequence = [EOS] + list(output_sequence) + [EOS]
    output_sequence = [char2int[c] for c in output_sequence]

    loss = []
    current = 0
    decoder = make_decoder(input_sequence)
    probs = next(decoder)  # get first decoding probabilities
    while len(loss) < len(output_sequence):
        char = output_sequence[current]
        loss.append(-dy.log(dy.pick(probs, char)))
        current += 1
        next(decoder)  # move decoder to expecting position (prev_char = yield)
        probs = decoder.send(char)  # provide decoder with actual char to produce next state
    loss = dy.esum(loss)
    return loss


def generate(input_sequence):
    def sample(probs):
        rnd = random.random()
        for i, p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    out = ''
    count_EOS = 0
    decoder = make_decoder(input_sequence)
    probs = next(decoder)  # get first decoding probabilities
    for i in range(len(input_sequence)*2):
        if count_EOS == 2: break
        char = sample(probs.vec_value())
        next(decoder)  # move decoder to expecting position
        probs = decoder.send(char)  # provide decoder with current sampled char to produce next state
        if int2char[char] == EOS:
            count_EOS += 1
            continue

        out += int2char[char]
    return out


def train(model, sentence):
    trainer = dy.SimpleSGDTrainer(model)
    for i in range(600):
        dy.renew_cg()
        loss = train_decode(sentence, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print(loss_value)
            print(generate(sentence))


train(model, "it is working")
