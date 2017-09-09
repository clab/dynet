import dynet as dy

train_sentence = [('the','D'), ('dog','N'), ('walks','V')]

i2w = dict(enumerate(set(w for w,t in train_sentence)))
w2i = dict((w,i) for i,w in i2w.items())
t2i = dict((t,i) for i,t in enumerate(set(t for w,t in train_sentence)))
num_words = len(w2i)
num_tags = len(t2i)

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

WEMB_DIM = 128
RNN_HIDDEN_DIM = 64
HIDDEN_DIM = 32

pWembs = model.add_lookup_parameters((num_words, WEMB_DIM))
pH = model.add_parameters((HIDDEN_DIM, RNN_HIDDEN_DIM))
pHb = model.add_parameters(HIDDEN_DIM)
pO = model.add_parameters((num_tags, HIDDEN_DIM))
pOb = model.add_parameters(num_tags)

rnn_builder = dy.BiRNNBuilder(1, WEMB_DIM, RNN_HIDDEN_DIM, model, dy.LSTMBuilder)


dy.renew_cg()

H = dy.parameter(pH)
Hb = dy.parameter(pHb)
O = dy.parameter(pO)
Ob = dy.parameter(pOb)

indexed_words, indexed_gold_tags = zip(*[(w2i[w], t2i[t]) for w,t in train_sentence]) 

wembs = [pWembs[wi] for wi in indexed_words]
noised_wembs = [dy.noise(we, 0.1) for we in wembs]

rnn_outputs = rnn_builder.transduce(noised_wembs)

errs = []
for rnn_output, gold_tag in zip(rnn_outputs, indexed_gold_tags):
    hidden = dy.tanh(dy.affine_transform([Hb, H, rnn_output]))
    model_tag = dy.affine_transform([Ob, O, hidden])
    err = dy.pickneglogsoftmax(model_tag, gold_tag)
    errs.append(err)
sum_errs = dy.esum(errs)

dy.print_graphviz(compact=False,
               show_dims=True,
                expression_names={ pWembs: "word_emb", 
                                   H: "H", Hb: "Hb",
                                   O: "O", Ob: "Ob", 
                                   wembs[0]: "first word"},
               lookup_names={"word_emb": i2w},
               collapse_birnns=True)

# Alternatively:
#   expression_names = dict((v,k) for (k,v) in dict(globals().items()+locals().items()).iteritems() if isinstance(v,Expression))
