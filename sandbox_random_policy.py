import gym_snakegame
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation, TimeLimit, AutoResetWrapper
from gym_snakegame.wrappers import RewardConverter
from gymnasium.experimental.wrappers import (
    ReshapeObservationV0,
    RecordVideoV0,
    LambdaRewardV0,
    DtypeObservationV0,
    LambdaObservationV0,
    LambdaRewardV0,
)
from tqdm import tqdm

env = gym.make(
    "gym_snakegame/SnakeGame-v0", board_size=15, n_target=1, render_mode="human"
)
env = ReshapeObservationV0(env, (1, 15, 15))
env = DtypeObservationV0(env, np.float32)
env = LambdaObservationV0(
    env,
    lambda obs: obs / env.ITEM,
    observation_space=gym.spaces.Box(
        0, 1, shape=(1, env.board_size, env.board_size), dtype=np.float32
    ),
)
env = LambdaRewardV0(env, lambda r: r * 5)
env = RewardConverter(env, -0.1)

observation, info = env.reset()
print(observation)
for i in range(1000000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
