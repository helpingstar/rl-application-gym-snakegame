import gym_snakegame
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import gym_snakegame
from gym_snakegame.wrappers import RewardConverter, SnakeActionMask
from gymnasium.wrappers import DtypeObservation, TransformObservation, TransformReward, ReshapeObservation, RecordVideo
from gymnasium.spaces import Box
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


n_channel = 4
board_size = 12
model_path = (
    "runs/gym_snakegame/SnakeGame-v0__ppo_v2_s12_action_mask__1__1709730835/cleanrl_ppo_v2_s12_action_mask_244140.pt"
)


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_channel, 32, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 6 * 6, 1024)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(1024, env.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(1024, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


env = gym.make("gym_snakegame/SnakeGame-v0", board_size=board_size, n_channel=4, render_mode="rgb_array")
env = RecordVideo(env, ".", episode_trigger=lambda x: x % 10 == 0, name_prefix="episode")
env = RecordVideo(env, ".", step_trigger=lambda x: x % 100 == 0, video_length=100, name_prefix="step")
env = DtypeObservation(env, np.float32)
env = TransformObservation(
    env, lambda obs: obs / env.unwrapped.ITEM, Box(0, 1, (n_channel, board_size, board_size), dtype=np.float32)
)
env = ReshapeObservation(env, (1, n_channel, board_size, board_size))
env = TransformReward(env, lambda r: r * 5)
env = RewardConverter(env, -0.01)
env = gym.wrappers.RecordEpisodeStatistics(env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(env).to(device)
agent.load_state_dict(torch.load(model_path))

obs, info = env.reset()
with torch.inference_mode():
    for i in range(100000):
        # print(torch.tensor(obs, dtype=int))
        # obs = torch.Tensor(obs).to(device)
        # action = agent.get_action_and_value(obs)[0]

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            obs, info = env.reset()
env.close()
