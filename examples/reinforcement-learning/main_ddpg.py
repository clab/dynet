import argparse
import time
from collections import deque

import os
import gym
import numpy as np
from ddpg import DDPG


def establish_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Walker2d-v2", type=str)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--memory_size", default=1e6, type=float)
    args = parser.parse_args()
    return args


def train():
    current_best_score = 0
    rewards = deque(maxlen=100)
    losses_actor = deque(maxlen=100)
    losses_critic = deque(maxlen=100)
    for _ in range(N_EPOCH):
        for _ in range(N_EPOCH_CYCLE):
            for _ in range(N_ROLLOUT):
                obs = env.reset()
                reward = 0
                for j in range(env._max_episode_steps):
                    action = player.act(obs)
                    obs_next, rwd, done, _ = env.step(action)
                    reward += rwd
                    player.store((obs, action, rwd, obs_next, 1. if done and j != env._max_episode_steps - 1 else 0))
                    if done: break
                    obs = obs_next

                rewards.append(reward)
            if player.learnable(BATCH_SIZE):
                for _ in range(N_TRAIN):
                    loss_actor, loss_critic = player.learn(BATCH_SIZE)
                    losses_actor.append(loss_actor)
                    losses_critic.append(loss_critic)

        print("==========================")
        print("last {} game mean value:".format(N_ROLLOUT))
        score = np.mean(rewards)
        if score > current_best_score:
            current_best_score = score
        if score > 1000:
            player.actor.pc.save("results/actor" + str(score))
            player.critic.pc.save("results/critic" + str(score))
        print("score {}, best score {}".format(score, current_best_score))
        print("noise {}".format(player.noise_stddev))
        print("Loss, Actor {}, Critic {}".format(np.mean(losses_actor), np.mean(losses_critic)))
        print("==========================\n")


def test(n_turns=10, name=None, render=False):
    player.noise_stddev = 0.05
    if name is not None:
        player.actor.pc.populate("results/actor" + name)
        player.critic.pc.populate("results/critic" + name)
    for i in range(n_turns):
        time.sleep(0.1)
        actions = []
        reward = 0
        obs = env.reset()
        for j in range(1000000):
            if render:
                env.render()
            action = player.act(obs)
            actions.append(action)
            obs, rwd, done, _ = env.step(action)
            reward += rwd
            if done:
                print("average action {}".format(np.mean(actions, axis=0)))
                print("reward {}".format(reward))
                break


args = establish_args()
env = gym.make(args.env_name)
BATCH_SIZE = 64
N_EPOCH = 500
N_EPOCH_CYCLE = 1
N_ROLLOUT = 100
N_TRAIN = 100
player = DDPG(env.observation_space.shape[0], env.action_space.shape[0], hiddens_actor=[64, 64],
              hiddens_critic=[64, 64], memory_size=int(args.memory_size))


if not os.path.exists("./results"):
    os.makedirs("./results")
if args.mode == "train":
    train()
    test()
else:
    test(name=args.model_name, render=args.render)
