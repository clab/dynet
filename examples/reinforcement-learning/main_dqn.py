import argparse
import gym

from dqn import DeepQNetwork
from network import MLP, Header
from train_test_utils import train_pipeline_progressive, train_pipeline_conservative, test


def establish_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynet-gpus', default=0, type=int)

    parser.add_argument('--env_id', default=0, type=int)
    parser.add_argument('--double', default=False, action='store_true')
    parser.add_argument('--dueling', default=False, action='store_true')
    parser.add_argument('--prioritized', default=False, action='store_true')
    args = parser.parse_args()
    return args


# ==== args ====
args = establish_args()

# ==== environment =====
ENVs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
score_thresholds = [499, -100, -100]

env_id = args.env_id
ENV, score_threshold = ENVs[env_id], score_thresholds[env_id]
env = gym.make(ENV)

# ==== DQN ====
MEMORY_SIZE = 50000

HIDDENS = [128]

network = Header(inpt_shape=env.observation_space.shape, hiddens=HIDDENS, opt_size=env.action_space.n,
                 network=MLP, dueling=args.dueling)
target_network = Header(inpt_shape=env.observation_space.shape, hiddens=HIDDENS,
                        opt_size=env.action_space.n, network=MLP, dueling=args.dueling) if args.double else None

dqn = DeepQNetwork(network=network, memory_size=MEMORY_SIZE, use_double_dqn=args.double, target_network=target_network,
                   dueling=args.dueling, prioritized=args.prioritized)

# ==== train & test ====
# choose one of the two pipelines
if env_id == 0:
    train_pipeline_conservative(env, dqn, score_threshold, n_epoch=500, n_rollout=100, n_train=1000, batch_size=256)
if env_id == 1 or env_id == 2:
    train_pipeline_progressive(env, dqn, score_threshold, batch_size=32, n_episode=2000)

test(env, dqn, n_turns=10, render=True)
