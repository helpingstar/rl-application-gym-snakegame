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


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_channel, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 1024),
            nn.ReLU(),
        )
        self.actor = nn.Linear(1024, env.action_space.n)
        self.critic = nn.Linear(1024, 1)

    def get_action(self, x, invalid_action_mask=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        action = probs.sample()
        return action


n_channel = 4
board_size = 12
model_path = (
    "runs/gym_snakegame/SnakeGame-v0__ppo_v2_s12_action_mask__1__1710016433/cleanrl_ppo_v2_s12_action_mask_244140.pt"
)
n_episode = 3

env = gym.make("gym_snakegame/SnakeGame-v0", board_size=board_size, n_channel=4, render_mode="rgb_array")
env = RecordVideo(env, ".", episode_trigger=lambda x: True, name_prefix="episode")
env = SnakeActionMask(env)
env = DtypeObservation(env, np.float32)
env = TransformObservation(
    env, lambda obs: obs / env.unwrapped.ITEM, Box(0, 1, (n_channel, board_size, board_size), dtype=np.float32)
)
env = ReshapeObservation(env, (1, n_channel, board_size, board_size))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent().to(device)
agent.load_state_dict(torch.load(model_path))

with torch.inference_mode():
    observation, info = env.reset()
    while n_episode > 0:
        invalid_action_mask = torch.Tensor(info["action_mask"]).to(device)
        action = agent.get_action(torch.Tensor(observation).to(device), invalid_action_mask=invalid_action_mask)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            n_episode -= 1
            if n_episode > 0:
                obs, info = env.reset()
env.close()
