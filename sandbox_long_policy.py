import gym_snakegame
import gymnasium as gym
import numpy as np
# env = gym.make('gym_snakegame/SnakeGame-v0', board_size=5, n_target=1, render_mode='human')
env = gym.make("MountainCar-v0")
print(env.observation_space)
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 100 == 0)
env = gym.experimental.wrappers.FrameStackObservationV0(env, 5)

observation, info = env.reset()

print(env.observation_space)

print(observation)
# print(env.observation_space)
for i in range(100000):
    # action = i % env.action_space.n
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    if terminated or truncated:
        env.reset()
env.close()