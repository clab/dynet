import numpy as np


def train_pipeline_progressive(env, player, score_threshold, batch_size, n_episode, learn_start=100, print_every=10):
    rewards, losses = [], []
    for i_episode in range(n_episode):
        obs = env.reset()
        reward = 0
        for t in range(env._max_episode_steps):
            action = player.act(obs)
            obs_next, rwd, done, _ = env.step(action)
            player.store((obs, action, rwd, obs_next, float(False if t == env._max_episode_steps - 1 else done)))
            reward += rwd
            if i_episode > learn_start:
                loss = player.learn(batch_size)
                if loss is not None: losses.append(loss)
            if done: break
            obs = obs_next
        rewards.append(reward)
        if i_episode % print_every == 0:
            score = np.mean(rewards[-100:])
            print("================================")
            print("i_episode: {}".format(i_episode))
            print("100 games mean reward: {}".format(score))
            if len(losses) > 0:
                print("100 games mean loss: {}".format(np.mean(losses[-100:])))
            print("================================")
            print()
            if score > score_threshold: break


def train_pipeline_conservative(env, player, score_threshold, batch_size, n_epoch, n_rollout, n_train, learn_start=0,
                                early_stop=True):
    for i_epoch in range(n_epoch):
        rewards, losses = [], []
        for _ in range(n_rollout):
            reward = 0
            obs = env.reset()
            for t in range(env._max_episode_steps):
                action = player.act(obs)
                obs_next, rwd, done, _ = env.step(action)
                player.store((obs, action, rwd, obs_next, float(False if t == env._max_episode_steps - 1 else done)))
                reward += rwd
                if done: break
                obs = obs_next
            rewards.append(reward)

        if i_epoch > learn_start:
            for _ in range(n_train):
                loss = player.learn(batch_size)
                if loss is not None: losses.append(loss)

        if i_epoch % 1 == 0:
            mean_reward = np.mean(rewards)
            print("===========================")
            print("i_epoch: {}".format(i_epoch))
            print("epsilon: {}".format(player.epsilon))
            print("Average score of {} rollout games: {}".format(n_rollout, mean_reward))
            if i_epoch > learn_start:
                print("Average training loss: {}".format(np.mean(losses)))
            print("===========================")
            print()
            if early_stop and mean_reward >= score_threshold: break


def test(env, player, n_turns, render=False):
    input("Ready to test., Press any key to coninue...")
    for _ in range(n_turns):
        score = 0
        obs = env.reset()
        for _ in range(env._max_episode_steps):
            if render: env.render()
            action = player.act(obs)
            obs, reward, done, _ = env.step(action)
            if render: env.render()
            score += reward
            if done:
                print('The score is {}'.format(score))
                break
            if render: env.render()
