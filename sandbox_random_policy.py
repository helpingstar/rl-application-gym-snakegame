import gym_snakegame
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation, TimeLimit, AutoResetWrapper
from gym_snakegame.wrappers import RewardConverter
from gymnasium.experimental.wrappers import ReshapeObservationV0, RecordVideoV0, LambdaRewardV0, DtypeObservationV0, LambdaObservationV0
from tqdm import tqdm

env = gym.make('gym_snakegame/SnakeGame-v0', board_size=9, n_target=1, render_mode=None)
# env = DtypeObservationV0(env, np.float32)
# env = ReshapeObservationV0(env, (1, 5, 5))
# env = LambdaObservationV0(env, lambda obs: obs / env.ITEM, observation_space=gym.spaces.Box(0, 1, shape=(env.board_size, env.board_size), dtype=np.float32))
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 100 == 0)

observation, info = env.reset()
print(observation)
for i in tqdm(range(1000000)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    a, b = np.unique(obs, return_counts=True)
    if len(a[b>2]) > 1:
        print(obs)
        print()
    if 1 not in obs:
        print(obs)
        print()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()