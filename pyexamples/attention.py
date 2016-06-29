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

enc_fwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
enc_bwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

dec_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2, STATE_SIZE, model)

lookup = model.add_lookup_parameters( (VOCAB_SIZE, EMBEDDINGS_SIZE))
attention_w = model.add_parameters( (1, STATE_SIZE*2+STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
decoder_b = model.add_parameters( (VOCAB_SIZE))


def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]

    global lookup

    return [lookup[char] for char in sentence]


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
    vectors = [pc.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(vectors, state):
    global attention_w
    w = pc.parameter(attention_w)
    attention_weights = []
    for vector in vectors:
        #concatenate each encoded vector with the current decoder state
        attention_input = pc.concatenate([vector, pc.concatenate(list(state.s()))])
        #get the attention wieght for the decoded vector
        attention_weights.append(w * attention_input)
    #normalize the weights
    attention_weights = pc.softmax(pc.concatenate(attention_weights))
    #apply the weights
    vectors = pc.esum([vector*attention_weight for vector, attention_weight in zip(vectors, attention_weights)])
    return vectors


def decode(dec_lstm, vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = pc.parameter(decoder_w)
    b = pc.parameter(decoder_b)


    s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE*2))

    loss = []
    for char in output:
        vector = attend(vectors, s)

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = pc.softmax(out_vector)
        loss.append(-pc.log(pc.pick(probs, char)))
    loss = pc.esum(loss)
    return loss


def generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    def sample(probs):
        rnd = random.random()
        for i, p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    embedded = embed_sentence(input)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = pc.parameter(decoder_w)
    b = pc.parameter(decoder_b)

    s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE * 2))
    out = ''
    count_EOS = 0
    for i in range(len(input)*2):
        if count_EOS == 2: break
        vector = attend(encoded, s)

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


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    pc.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)


def train(model, sentence):
    trainer = pc.SimpleSGDTrainer(model)
    for i in xrange(400):
        loss = get_loss(sentence, sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print loss_value
            print generate(sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)


train(model, "it is working")


