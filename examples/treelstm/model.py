import dynet as dy
import pickle
import os


class TreeLSTMBuilder(object):
    def __init__(self, pc_param, pc_embed, word_vocab, wdim, hdim, word_embed=None, dropout_rate=0.5):
        self.WS = [pc_param.add_parameters((hdim, wdim)) for _ in "iou"]
        self.US = [pc_param.add_parameters((hdim, 2 * hdim)) for _ in "iou"]
        self.UFS = [pc_param.add_parameters((hdim, 2 * hdim)) for _ in "ff"]
        self.BS = [pc_param.add_parameters(hdim) for _ in "iouf"]
        self.E = pc_embed.add_lookup_parameters((len(word_vocab), wdim), init=word_embed)
        self.w2i = word_vocab
        self.use_dropout = dropout_rate > 0
        self.dropout_rate = dropout_rate

    def expr_for_tree(self, tree, decorate=False, training=True):
        assert (not tree.isleaf())
        if len(tree.children) == 1:
            assert (tree.children[0].isleaf())
            emb = self.E[self.w2i.get(tree.children[0].label, 0)]
            Wi, Wo, Wu = [dy.parameter(w) for w in self.WS]
            bi, bo, bu, _ = [dy.parameter(b) for b in self.BS]
            i = dy.logistic(dy.affine_transform([bi, Wi, emb]))
            o = dy.logistic(dy.affine_transform([bo, Wo, emb]))
            u = dy.tanh(dy.affine_transform([bu, Wu, emb]))
            if self.use_dropout and training:
                u = dy.dropout(u, self.dropout_rate)
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
            if decorate: tree._e = h
            return h, c
        assert (len(tree.children) == 2), tree.children[0]
        e1, c1 = self.expr_for_tree(tree.children[0], decorate)
        e2, c2 = self.expr_for_tree(tree.children[1], decorate)
        Ui, Uo, Uu = [dy.parameter(u) for u in self.US]
        Uf1, Uf2 = [dy.parameter(u) for u in self.UFS]
        bi, bo, bu, bf = [dy.parameter(b) for b in self.BS]
        e = dy.concatenate([e1, e2])
        i = dy.logistic(dy.affine_transform([bi, Ui, e]))
        o = dy.logistic(dy.affine_transform([bo, Uo, e]))
        f1 = dy.logistic(dy.affine_transform([bf, Uf1, e]))
        f2 = dy.logistic(dy.affine_transform([bf, Uf2, e]))
        u = dy.tanh(dy.affine_transform([bu, Uu, e]))
        if self.use_dropout and training:
            u = dy.dropout(u, self.dropout_rate)
        c = dy.cmult(i, u) + dy.cmult(f1, c1) + dy.cmult(f2, c2)
        h = dy.cmult(o, dy.tanh(c))
        if decorate: tree._e = h
        return h, c


class TreeLSTMClassifier(object):
    def __init__(self, n_classes, w2i, word_embed, params, model_meta_file=None):
        self.params = params.copy()
        if model_meta_file is not None:
            with open(model_meta_file, 'rb') as f:
                saved_params = pickle.load(f)
            self.params.update(saved_params)
        self.pc_param = dy.ParameterCollection()
        self.pc_embed = dy.ParameterCollection()
        self.builder = TreeLSTMBuilder(self.pc_param, self.pc_embed, w2i, self.params['wembed_size'], self.params['hidden_size'], word_embed, self.params['dropout_rate'])
        self.W_ = self.pc_param.add_parameters((n_classes, self.params['hidden_size']))
        if model_meta_file is not None:
            self._load_param_embed(model_meta_file)

    def predict_for_tree(self, tree, decorate=True, training=True):
        h, c = self.builder.expr_for_tree(tree, decorate, training)
        W = dy.parameter(self.W_)
        if decorate:
            for node in tree.nonterms_iter():
                node._logits = W * node._e
        else:
            tree._logits = W * h
        if not training: return tree._logits.npvalue()

    def losses_for_tree(self, tree, sum=True):
        self.predict_for_tree(tree, decorate=True, training=True)
        nodes = tree.nonterms()
        losses = [dy.pickneglogsoftmax(nt._logits, nt.label) for nt in nodes]
        return dy.esum(losses) if sum else losses, len(nodes)

    def losses_for_tree_batch(self, trees):
        batch_losses = []
        for tree in trees:
            losses, _ = self.losses_for_tree(tree, sum=False)
            batch_losses += losses
        return dy.esum(batch_losses) / len(batch_losses)

    def regularization_loss(self, coef):
        losses = [dy.l2_norm(p) ** 2 for p in self.pc_param.parameters_list()]
        return (coef / 2) * dy.esum(losses)

    def save(self, save_dir, model_name):
        meta_path = os.path.join(save_dir, 'meta', model_name)
        param_path = os.path.join(save_dir, 'param', model_name)
        embed_path = os.path.join(save_dir, 'embed', model_name)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.params, f)
        self.pc_param.save(param_path)
        self.pc_embed.save(embed_path)
        return meta_path
    
    def _load_param_embed(self, model_meta_file):
        param_path = model_meta_file.replace('meta', 'param')
        embed_path = model_meta_file.replace('meta', 'embed')
        self.pc_param.populate(param_path)
        self.pc_embed.populate(embed_path)

    def delete(self, model_meta_file):
        if model_meta_file is None: return
        os.remove(model_meta_file)
        os.remove(model_meta_file.replace('meta', 'param'))
        os.remove(model_meta_file.replace('meta', 'embed'))
