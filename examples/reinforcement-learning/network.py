import dynet as dy


class Network(object):
    def __init__(self, pc):
        self.pc = dy.ParameterCollection() if pc is None else pc

    def update(self, other, soft=False, tau=0.1):
        params_self, params_other = self.pc.parameters_list(), other.pc.parameters_list()
        for x, y in zip(params_self, params_other):
            target_values = ((1 - tau) * x.as_array() + tau * y.as_array()) if soft else y.as_array()
            x.set_value(target_values)


class MLP(Network):
    def __init__(self, inpt_shape, hiddens, activation=dy.rectify, layer_norm=False, pc=None):
        super().__init__(pc)
        if len(inpt_shape) != 1:
            raise ValueError("inpt_shape must be 1 dimension for MLP.")
        self.specified_activation = hasattr(activation, "__len__")
        self.activation = activation
        self.layer_norm = layer_norm
        units = [inpt_shape[0]] + hiddens
        self.Ws, self.bs = [], []
        if layer_norm:
            self.ln_gs, self.ln_bs = [], []
        for i in range(len(units) - 1):
            self.Ws.append(self.pc.add_parameters((units[i + 1], units[i])))
            self.bs.append(self.pc.add_parameters(units[i + 1]))
            if layer_norm:
                self.ln_gs.append(self.pc.add_parameters(units[i + 1]))
                self.ln_bs.append(self.pc.add_parameters(units[i + 1]))
        self.n_layers = len(self.Ws)

    def __call__(self, obs, batched=False):
        out = obs if isinstance(obs, dy.Expression) else dy.inputTensor(obs, batched=batched)

        for i in range(self.n_layers):
            b, W = dy.parameter(self.bs[i]), dy.parameter(self.Ws[i])
            out = dy.affine_transform([b, W, out])
            if self.layer_norm and i != self.n_layers - 1:
                out = dy.layer_norm(out, self.ln_gs[i], self.ln_bs[i])
            if self.specified_activation:
                if self.activation[i] is not None:
                    out = self.activation[i](out)
            else:
                out = self.activation(out)
        return out


class Header(Network):
    def __init__(self, opt_size, network, dueling=False, **kwargs):
        super().__init__(None)
        self.network = network(**kwargs, pc=self.pc)
        self.opt_size = opt_size
        self.dueling = dueling

        hiddens = kwargs['hiddens']
        self.W = self.pc.add_parameters((opt_size, hiddens[-1]))
        self.b = self.pc.add_parameters(opt_size)
        if dueling:
            self.W_extra = self.pc.add_parameters((1, hiddens[-1]))
            self.b_extra = self.pc.add_parameters(1)

    def __call__(self, obs, batched=False):
        out = self.network(obs, batched)
        W, b = dy.parameter(self.W), dy.parameter(self.b)
        As = dy.affine_transform([b, W, out])
        if self.dueling:
            W_extra, b_extra = dy.parameter(self.W_extra), dy.parameter(self.b_extra)
            V = dy.affine_transform([b_extra, W_extra, out])
            return As, V
        return As

