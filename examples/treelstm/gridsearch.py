from pydoc import locate


class GridSearch:
    def __init__(self, filename):
        self.param_names = []
        self.param_values = []
        self.in_iteration = False
        with open(filename, 'r') as f:
            for line in f:
                param_name, param_type, param_values = line.strip().split(' ')
                param_type = locate(param_type)
                param_values = [param_type(str_value) for str_value in param_values.split(',')]
                self._add(param_name, param_values)
        self.reset()

    def _add(self, param_name, param_value):
        if self.in_iteration:
            raise RuntimeError('Can not add search parameter in iteration.')
        self.param_names.append(param_name)
        self.param_values.append(param_value)

    def __iter__(self):
        while self._has_next:
            yield self._next_group()
            self._add_iter()

    def reset(self):
        self._start_iteration()

    def _next_group(self):
        param_dict = {}
        for i, idx in enumerate(self.iter):
            param_dict[self.param_names[i]] = self.param_values[i][idx]
        return param_dict

    def _add_iter(self):
        self._has_next = False
        for i in range(len(self.upper_bound)):
            self.iter[i] += 1
            if self.iter[i] < self.upper_bound[i]:
                self._has_next = True
                break
            else:
                self.iter[i] -= self.upper_bound[i]

    def _start_iteration(self):
        self.in_iteration = True
        self._has_next = True if len(self.param_names) > 0 else False
        self.upper_bound = [len(p) for p in self.param_values]
        self.iter = [0] * len(self.param_names)


