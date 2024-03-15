# rl-application-gym-snakegame

This is the code to solve [**gym-snakegame**](https://github.com/helpingstar/gym-snakegame), a reinforcement learning environment based on the 2048 game.

An action_mask is code that explicitly masks situations where the game terminates by hitting itself or hitting a wall. This is different from masking invalid actions.

`*_action_mask.py` uses the [`SnakeActionMask`](https://github.com/helpingstar/gym-snakegame/blob/main/gym_snakegame/wrappers/snake_action_mask.py) wrapper. 

Not masking an action doesn't necessarily reduce performance, and performance can vary depending on conditions in the environment, such as the size of the board, or the reward design.

Detailed experimental results can be found in the wandb project link below.
* https://wandb.ai/iamhelpingstar/snakegame?nw=nwuseriamhelpingstar

After training, you can modify the `weight_path` in the `monitoring.py` file to record the agent's gameplay.

The parameters for training are inside the `ppo_v2_s12*.py` file and can be modified as desired.

The code referenced
* https://github.com/vwxyzjn/cleanrl
* https://github.com/vwxyzjn/invalid-action-masking


![snake_length](/figure/snake_length.png)

![episodic_length](/figure/episodic_length.png)

![episodic_return](/figure/episodic_return.png)

