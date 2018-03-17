import codecs
import numpy as np
import dynet as dy


def acc_eval(dataset, model):
    dataset.reset(shuffle=False)
    good = bad = 0.0
    for tree in dataset:
        dy.renew_cg()
        pred = np.argmax(model.predict_for_tree(tree, decorate=False, training=False))
        if pred == tree.label:
            good += 1
        else:
            bad += 1
    acc = good / (good + bad)
    return acc


def get_embeds(embed_path):
    word_embeds, w2i = [np.random.randn(300)], {'_UNK_': 0}
    with codecs.open(embed_path) as f:
        for line in f:
            line = line.strip().split(' ')
            word, embed = line[0], line[1:]
            w2i[word] = len(word_embeds)
            word_embeds.append(np.array(embed, dtype=np.float32))
    w2i['-LRB-'] = w2i['(']
    w2i['-RRB-'] = w2i[')']
    return np.array(word_embeds), w2i
