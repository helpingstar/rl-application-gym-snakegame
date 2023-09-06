import gym_snakegame
import gymnasium as gym

from gymnasium.wrappers import TransformObservation

env = gym.make('gym_snakegame/SnakeGame-v0', size=5, n_target=1, render_mode='human')
env = TransformObservation(env, lambda obs: obs / 5.0)
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 200 == 0)

observation, info = env.reset()
print(env.observation_space.shape)
print(env.action_space)
print(env.size)
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        env.reset()
env.close()