import gym_snakegame
import gymnasium as gym

from gymnasium.wrappers import TransformObservation, TimeLimit, AutoResetWrapper
from gym_snakegame.wrappers import RewardConverter
from gymnasium.experimental.wrappers import ReshapeObservationV0, RecordVideoV0, LambdaRewardV0

env = gym.make('gym_snakegame/SnakeGame-v0', board_size=15, n_target=1, render_mode='human')
env = LambdaRewardV0(env, lambda r: r * 10.0)
env = TimeLimit(env, 10000)
env = ReshapeObservationV0(env, (1, env.board_size, env.board_size))
env = TransformObservation(env, lambda obs: obs / 5.0)
env = RewardConverter(env, -0.01)
env = AutoResetWrapper(env)
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 100 == 0)

observation, info = env.reset()
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if reward != -0.01:
        print(reward)
    if 'episode' in info:
        print(info['episode'])
env.close()