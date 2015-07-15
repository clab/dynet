from pycnn import *

LAYERS = 3
INPUT_DIM = 500
HIDDEN_DIM = 500
INPUT_VOCAB_SIZE = 0
OUTPUT_VOCAB_SIZE = 0

class EncoderDecoder:
    def __init__(self, model, builder=LSTMBuilder):
        self.m = model
        model.add_parameters("ie2h", (HIDDEN_DIM*LAYERS*1.5, HIDDEN_DIM*LAYERS*2))
        model.add_parameters("bie", HIDDEN_DIM*LAYERS*1.5)
        model.add_parameters("h2oe", (HIDDEN_DIM*LAYERS, HIDDEN_DIM*LAYERS*1.5))
        model.add_parameters("boe", (HIDDEN_DIM*LAYERS))
        model.add_lookup_parameters("c", (INPUT_VOCAB_SIZE, INPUT_DIM))
        model.add_lookup_parameters("ec", (INPUT_VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (OUTPUT_VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", OUTPUT_VOCAB_SIZE)

        self.dec_builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
        self.fwd_enc_builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
        self.rev_enc_builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

    def BuildGraph(self, isent, osent, cg):
        fwb = self.fwd_enc_builder
        fwb.new_graph(cg)
        fwb.start_new_sequence()
        # fw encoder
        for wid in isent:
            fwb.add_input(cg.lookup(model["ec"], wid))
        # bw encoder
        bwb = self.rev_enc_builder
        bwb.new_graph(cg)
        bwb.start_new_sequence()
        for wid in reversed(isent):
            bwb.add_input(cg.lookup(model["ec"], wid))

        # combine encoders:
        to = fwb.final_h() + bwb.final_h()
        combined = concatenate(to)
        ie2h = cg.parameters(self.m["ie2h"])
        bie  = cg.parameters(self.m["bie"])
        t = bie + (ie2h * combined)
        cg.inc_forward_vec()
        h = rectify(t)
        h2oe = cg.parameters(self.m["h2oe"])
        boe  = cg.parameters(self.m["boe"])
        nc = boe + (h2oe * h)

        oein1 = [pickrange(nc, i* HIDDEN_DIM, (i+1)*HIDDEN_DIM) for i in xrange(LAYERS)]
        oein2 = [tanh(x) for x in oein1]
        oein = oein1 + oein2

        decb = self.dec_builder
        decb.new_graph(cg)
        decb.start_new_sequence(oein)
        # decoder
        R = cg.parameters(self.m["R"])
        bias = cg.parameters(self.m["bias"])

        errs = []
        for (cw,nw) in zip(osent, osent[1:]):
            x_t = cg.lookup(self.m["c"], cw)
            y_t = decb.add_input(x_t)
            r_t = bias + (R * y_t)
            ydist = log_softmax(r_t)
            errs.append(cg.outputPicker(ydist, nw))
        return -esum(errs)

if __name__ == '__main__':
    import sys
    import util
    import random

    train = util.CharsCorpusReader(sys.argv[1],begin="<s>")
    vocab = util.Vocab.from_corpus(train)
    INPUT_VOCAB_SIZE = vocab.size()
    OUTPUT_VOCAB_SIZE = vocab.size()

    model = Model()
    sgd = SimpleSGDTrainer(model)
    cg = ComputationGraph()

    lm = EncoderDecoder(model, LSTMBuilder)

    train = list(train)
    chars = loss = 0.0
    while True:
        random.shuffle(train)
        for i,sent in enumerate(train,1):
            if i % 50 == 0:
                print "E = ", (loss / chars)
                sgd.status()

                chars = loss = 0.0

            cg.renew()
            chars += len(sent) - 1
            isent = [vocab.w2i[w] for w in sent]
            lm.BuildGraph(isent, isent, cg)
            loss += cg.forward_scalar()
            cg.backward()
            sgd.update()
        sgd.update_epoch()

