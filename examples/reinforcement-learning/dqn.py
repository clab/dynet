import dynet as dy
import numpy as np

from memory import Memory, PrioritizedMemory


# DeepQNetwork: https://arxiv.org/abs/1312.5602
# An reinforcement learning agent to learn in environments which have discrete action spaces.
# Double Q-Learning: https://arxiv.org/abs/1509.06461
# Prioritized Replay: https://arxiv.org/abs/1511.05952
# Dueling Network Architectures: https://arxiv.org/abs/1511.06581
class DeepQNetwork(object):
    def __init__(self, network, memory_size, use_double_dqn=False, target_network=None, n_replace_target=500,
                 dueling=True, prioritized=True):
        self.network = network
        self.trainer = dy.AdamTrainer(network.pc)
        self.trainer.set_clip_threshold(1.)
        self.trainer.set_learning_rate(5e-4)
        self.epsilon = 1.
        self.epsilon_decrease = 1e-4
        self.epsilon_lower = 0.05
        self.reward_decay = 0.99
        self.learn_step = 0

        self.use_double_dqn = use_double_dqn
        if use_double_dqn:
            self.target_network = target_network
            self.n_replace_target = n_replace_target
            self.target_network.update(network)
        self.dueling = dueling
        self.prioritized = prioritized
        if prioritized:
            self.beta = 0.
            self.beta_increase = self.epsilon_decrease

        self.memory = PrioritizedMemory(memory_size) if prioritized else Memory(memory_size)

    def act(self, obs, deterministic=True):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.network.opt_size)
        dy.renew_cg()
        if self.dueling:
            actions, _ = self.network(obs)
            actions = actions.npvalue()
        else:
            actions = self.network(obs).npvalue()

        if deterministic:
            return np.argmax(actions)
        else:
            return np.random.choice(self.network.opt_size, p=actions)

    def store(self, exp):
        self.memory.store(exp)

    def learn(self, batch_size):
        if self.prioritized:
            if not self.memory.is_full(): return -np.inf
            indices, exps, weights = self.memory.sample(batch_size, self.beta)
        else:
            exps = self.memory.sample(batch_size)
        obss, actions, rewards, obs_nexts, dones = self._process(exps)

        dy.renew_cg()
        target_network = self.target_network if self.use_double_dqn else self.network
        if self.dueling:
            target_values, v = target_network(obs_nexts, batched=True)
            target_values = target_values.npvalue() + v.npvalue()
        else:
            target_values = target_network(obs_nexts, batched=True)
            target_values = target_values.npvalue()
        target_values = np.max(target_values, axis=0)
        target_values = rewards + self.reward_decay * (target_values * (1 - dones))

        dy.renew_cg()
        if self.dueling:
            all_values_expr, v = self.network(obss, batched=True)
        else:
            all_values_expr = self.network(obss, batched=True)
        picked_values = dy.pick_batch(all_values_expr, actions)
        diff = (picked_values + v if self.dueling else picked_values) - dy.inputTensor(target_values, batched=True)
        if self.prioritized:
            self.memory.update(indices, np.transpose(np.abs(diff.npvalue())))
        losses = dy.pow(diff, dy.constant(1, 2))
        if self.prioritized:
            losses = dy.cmult(losses, dy.inputTensor(weights, batched=True))
        loss = dy.sum_batches(losses)
        loss_value = loss.npvalue()
        loss.backward()
        self.trainer.update()

        self.epsilon = max(self.epsilon - self.epsilon_decrease, self.epsilon_lower)
        if self.prioritized:
            self.beta = min(self.beta + self.beta_increase, 1.)

        self.learn_step += 1
        if self.use_double_dqn and self.learn_step % self.n_replace_target == 0:
            self.target_network.update(self.network)
        return loss_value

    # data in memory: [N, exp], exp: [obs, action, reward, obs_next, done]
    # output: [obss, actions, rewards, obs_nexts, dones], 'X's: [x, batch_size]
    @staticmethod
    def _process(exps):
        n = len(exps)
        ret = []
        for i in range(5):
            ret.append([])
            for j in range(n):
                ret[i].append(exps[j][i])

        ret = [np.transpose(arr) for arr in ret]
        return ret
