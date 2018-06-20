import dynet as dy
import numpy as np

from memory import Memory
from network import MLP


# Deep Deterministic Policy Gradient: https://arxiv.org/abs/1509.02971
# An reinforcement learning agent to learn in environments which have continuous action spaces.
class DDPG:
    def __init__(self, obs_dim, action_dim, hiddens_actor, hiddens_critic, layer_norm=False, memory_size=50000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.noise_stddev = 1.
        self.noise_stddev_decrease = 5e-4
        self.noise_stddev_lower = 5e-2

        actor_activations = [dy.tanh for _ in range(len(hiddens_actor))] + [dy.tanh]
        critic_activations = [dy.tanh for _ in range(len(hiddens_critic))] + [None]
        self.actor = MLP(inpt_shape=(obs_dim,), hiddens=hiddens_actor + [action_dim], activation=actor_activations,
                         layer_norm=layer_norm)
        self.critic = MLP(inpt_shape=(obs_dim + action_dim,), hiddens=hiddens_critic + [1],
                          activation=critic_activations, layer_norm=layer_norm)
        self.actor_target = MLP(inpt_shape=(obs_dim,), hiddens=hiddens_actor + [action_dim],
                                activation=actor_activations, layer_norm=layer_norm)
        self.critic_target = MLP(inpt_shape=(obs_dim + action_dim,), hiddens=hiddens_critic + [1],
                                 activation=critic_activations, layer_norm=layer_norm)
        self.actor_target.update(self.actor, soft=False)
        self.critic_target.update(self.critic, soft=False)

        self.trainer_actor = dy.AdamTrainer(self.actor.pc)
        self.trainer_critic = dy.AdamTrainer(self.critic.pc)
        self.trainer_actor.set_learning_rate(1e-4)
        self.trainer_critic.set_learning_rate(1e-3)

        self.memory = Memory(memory_size)

    def act(self, obs):
        dy.renew_cg()
        action = self.actor(obs).npvalue()
        if self.noise_stddev > 0:
            noise = np.random.randn(self.action_dim) * self.noise_stddev
            action += noise
        return np.clip(action, -1, 1)

    def store(self, exp):
        self.memory.store(exp)

    def learn(self, batch_size):
        exps = self.memory.sample(batch_size)
        obss, actions, rewards, obs_nexts, dones = self._process(exps)

        # Update critic
        dy.renew_cg()
        target_actions = self.actor_target(obs_nexts, batched=True)
        target_values = self.critic_target(dy.concatenate([dy.inputTensor(obs_nexts, batched=True), target_actions]),
                                           batched=True)
        target_values = rewards + 0.99 * target_values.npvalue() * (1 - dones)

        dy.renew_cg()
        values = self.critic(np.concatenate([obss, actions]), batched=True)
        loss = dy.mean_batches((values - dy.inputTensor(target_values, batched=True)) ** 2)
        loss_value_critic = loss.npvalue()
        loss.backward()
        self.trainer_critic.update()

        # update actor
        dy.renew_cg()
        actions = self.actor(obss, batched=True)
        obs_and_actions = dy.concatenate([dy.inputTensor(obss, batched=True), actions])
        loss = -dy.mean_batches(self.critic(obs_and_actions, batched=True))
        loss_value_actor = loss.npvalue()
        loss.backward()
        self.trainer_actor.update()

        self.noise_stddev = (
                    self.noise_stddev - self.noise_stddev_decrease) if self.noise_stddev > self.noise_stddev_lower else self.noise_stddev_lower

        self.actor_target.update(self.actor, soft=True)
        self.critic_target.update(self.critic, soft=True)

        return loss_value_actor + loss_value_critic

    # data in memory: [memory_size, exp], exp: [obs, action, reward, obs_next, done]
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

    @property
    def epsilon(self):
        return self.noise_stddev
