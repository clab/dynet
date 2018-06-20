import argparse
import gym
from ddpg import DDPG
from train_test_utils import train_pipeline_conservative, test


def establish_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Walker2d-v2", type=str)
    parser.add_argument("--memory_size", default=1e6, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_epoch", default=500, type=int)
    parser.add_argument("--rollout_per_epoch", default=100, type=int)
    parser.add_argument("--train_per_epoch", default=100, type=int)
    args = parser.parse_args()
    return args


args = establish_args()
env = gym.make(args.env_name)
player = DDPG(env.observation_space.shape[0], env.action_space.shape[0], hiddens_actor=[64, 64],
              hiddens_critic=[64, 64], memory_size=int(args.memory_size))

train_pipeline_conservative(env, player, score_threshold=999, batch_size=args.batch_size, n_epoch=args.n_epoch,
                            n_rollout=args.rollout_per_epoch, n_train=args.train_per_epoch)
test(env, player, n_turns=10, render=True)
