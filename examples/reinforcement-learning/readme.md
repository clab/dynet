# DeepQNetwork and DDPG
This is a dynet implementation of two reinforcement learning algorithms, [DeepQNetwork](https://arxiv.org/abs/1312.5602) and [DDPG](https://arxiv.org/abs/1509.02971). Some techniques in DeepQNetwork: [double Q learning](https://arxiv.org/abs/1509.06461), [prioritized replay](https://arxiv.org/abs/1511.05952) and [dueling network architectures](https://arxiv.org/abs/1511.06581) are included.

# Results
## Openai Gym classical control (Discrete action space with DeepQNetwork):
<img src="https://github.com/zhiyong1997/resources/blob/master/gifs/Cartpole.gif" width="200"> <img src="https://github.com/zhiyong1997/resources/blob/master/gifs/MountainCar.gif" width="200"> <img src="https://github.com/zhiyong1997/resources/blob/master/gifs/Acrobot.gif" width="200">


## Openai Gym mujoco (Continous control with DDPG):
<img src="https://github.com/zhiyong1997/resources/blob/master/gifs/stand_still.gif" width="200"> <img src="https://github.com/zhiyong1997/resources/blob/master/gifs/move.gif" width="200"> <img src="https://github.com/zhiyong1997/resources/blob/master/gifs/hopforward.gif" width="200">

The two-leg robot is supposed not to fall down and to move forward as fast as possible. The above three different strategies learned by the bot are 'standing still', 'moving slowly' and 'hopping forward'. The corresponding reward of these strategies in the 'Walker2d-v2' environment is around 1050, 1200 and 1600, respectively. (Random policy gets 1.7 in average.)

# Play with the code
## DeepQNetwork
```
python main_dqn.py --env_id 0
```
or
```
python main_dqn.py --env_id 0 --double --dueling --prioritized
```

## DDPG
```
python main_ddpg.py
```