import gym_snakegame
import gymnasium as gym

env = gym.make('gym_snakegame/SnakeGame-v0', board_size=15, n_target=1, render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 100 == 0)

observation, info = env.reset()
for i in range(100000):
    action = i % env.action_space.n
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
env.close()