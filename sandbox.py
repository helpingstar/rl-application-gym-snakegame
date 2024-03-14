import gym_snakegame
import gymnasium as gym
import numpy as np

import gym_snakegame
from gym_snakegame.wrappers import RewardConverter, SnakeActionMask
from gymnasium.wrappers import DtypeObservation, TransformObservation, TransformReward
from gymnasium.spaces import Box

env = gym.make("gym_snakegame/SnakeGame-v0", board_size=5, n_channel=4, render_mode="human")
env = SnakeActionMask(env)
env = DtypeObservation(env, np.float32)
env = TransformObservation(env, lambda obs: obs / env.unwrapped.ITEM, Box(0, 1, (4, 5, 5), dtype=np.float32))
env = TransformReward(env, lambda r: r * 5)
env = RewardConverter(env, -0.01)

env = gym.wrappers.RecordEpisodeStatistics(env)

print(env.observation_space)

observation, info = env.reset()

for i in range(100000):
    action = env.action_space.sample(info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
