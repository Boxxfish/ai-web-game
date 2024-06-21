from argparse import ArgumentParser
from dataclasses import dataclass
from functools import reduce
from typing import *

import gymnasium as gym
import torch
import torch.nn as nn
from torch import Tensor
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from torch.distributions import Categorical
from tqdm import tqdm

from webgame.algorithms.ppo import train_ppo
from webgame.algorithms.rollout_buffer import RolloutBuffer
from webgame.common import process_obs
from webgame.conf import entity
from webgame.envs import MAX_OBJS, OBJ_DIM, GameEnv
from webgame.train_filter import Backbone

_: Any


@dataclass
class Config:
    num_envs: int = (
        256  # Number of environments to step through at once during sampling.
    )
    train_steps: int = (
        128  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
    )
    iterations: int = 1000  # Number of sample/train iterations.
    train_iters: int = 2  # Number of passes over the samples collected.
    train_batch_size: int = 512  # Minibatch size while training models.
    discount: float = 0.98  # Discount factor applied to rewards.
    lambda_: float = 0.95  # Lambda for GAE.
    epsilon: float = 0.2  # Epsilon for importance sample clipping.
    max_eval_steps: int = 500  # Number of eval runs to average over.
    eval_steps: int = 8  # Max number of steps to take during each eval run.
    v_lr: float = 0.01  # Learning rate of the value net.
    p_lr: float = 0.001  # Learning rate of the policy net.
    use_objs: bool = False  # Whether we should use objects in the simulation.
    use_pos: bool = False  # Whether we use a position encoding.
    device: str = "cuda"  # Device to use during training.


class ValueNet(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        proj_dim = 32
        self.backbone = Backbone(channels, proj_dim, size, use_pos, objs_shape)
        self.net = nn.Sequential(
            nn.Conv2d(proj_dim, 32, 3, padding="same", dtype=torch.float),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding="same", dtype=torch.float),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(size**2 * 16, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Optional[Tensor],  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Optional[Tensor],  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:
        features = self.backbone(grid, objs, objs_attn_mask)
        values = self.net(features)
        return values


class PolicyNet(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
        action_count: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        proj_dim = 32
        self.backbone = Backbone(channels, proj_dim, size, use_pos, objs_shape)
        self.net = nn.Sequential(
            nn.Conv2d(proj_dim, 32, 3, padding="same", dtype=torch.float),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding="same", dtype=torch.float),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(size**2 * 16, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_count),
        )

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Optional[Tensor],  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Optional[Tensor],  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:
        features = self.backbone(grid, objs, objs_attn_mask)
        values = self.net(features)
        return values


class AgentData:
    def __init__(
        self,
        channels: int,
        grid_size: int,
        max_objs: int,
        obj_dim: int,
        cfg: Config,
        act_count: int,
    ):
        self.v_net = ValueNet(
            channels,
            grid_size,
            cfg.use_pos,
            (max_objs, obj_dim) if cfg.use_objs else None,
        )
        self.p_net = PolicyNet(
            channels,
            grid_size,
            act_count,
            cfg.use_pos,
            (max_objs, obj_dim) if cfg.use_objs else None,
        )
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=cfg.v_lr)
        self.p_opt = torch.optim.Adam(self.p_net.parameters(), lr=cfg.p_lr)
        self.buffer = RolloutBuffer(
            [
                (torch.Size((channels, grid_size, grid_size)), torch.float),
                (torch.Size((max_objs, obj_dim)), torch.float),
                (torch.Size((max_objs,)), torch.bool),
            ],
            torch.Size((1,)),
            torch.Size((act_count,)),
            torch.int,
            cfg.num_envs,
            cfg.train_steps,
        )


if __name__ == "__main__":
    cfg = Config()
    parser = ArgumentParser()
    for k, v in cfg.__dict__.items():
        if isinstance(v, bool):
            parser.add_argument(
                f"--{k.replace('_', '-')}", default=v, type=type(v), action="store_true"
            )
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    args = parser.parse_args()
    cfg = Config(**args.__dict__)
    device = torch.device(cfg.device)

    wandb.init(
        project="pursuer",
        entity=entity,
        config=cfg.__dict__,
    )

    env = GameEnv(cfg.use_objs)
    test_env = GameEnv(cfg.use_objs)

    # Initialize policy and value networks
    channels = 7 + 2
    grid_size = 8
    max_objs = MAX_OBJS
    obj_dim = OBJ_DIM
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete)
    agents = {
        agent: AgentData(channels, grid_size, max_objs, obj_dim, cfg, int(act_space.n))
        for agent in env.agents
    }

    obs_ = env.reset()[0]
    obs = {agent: process_obs(obs_[agent]) for agent in env.agents}
    for _ in tqdm(range(cfg.iterations), position=0):
        # Collect experience for a number of steps and store it in the buffer
        with torch.no_grad():
            for _ in tqdm(range(cfg.train_steps), position=1):
                all_action_probs = {}
                all_actions = {}
                for agent in env.agents:
                    action_probs = agents[agent].p_net(obs)
                    actions = Categorical(logits=action_probs).sample().numpy()
                    all_action_probs[agent] = action_probs
                    all_actions[agent] = actions
                obs_, rewards, dones, truncs, _ = env.step(all_actions)
                for agent in env.agents:
                    agents[agent].buffer.insert_step(
                        obs,
                        torch.from_numpy(all_actions[agent]).unsqueeze(-1),
                        action_probs[agent],
                        rewards[agent],
                        dones[agent],
                        truncs[agent],
                    )
                obs = {agent: process_obs(obs_[agent]) for agent in env.agents}
            for agent in env.agents:
                agents[agent].buffer.insert_final_step(obs[agent])

        # Train
        log_dict = {}
        for agent in env.agents:
            total_p_loss, total_v_loss = train_ppo(
                agents[agent].p_net,
                agents[agent].v_net,
                agents[agent].p_opt,
                agents[agent].v_opt,
                agents[agent].buffer,
                device,
                cfg.train_iters,
                cfg.train_batch_size,
                cfg.discount,
                cfg.lambda_,
                cfg.epsilon,
            )
            agents[agent].buffer.clear()
            log_dict[f"{agent}_avg_v_loss"] = total_v_loss / cfg.train_iters
            log_dict[f"{agent}_avg_p_loss"] = total_p_loss / cfg.train_iters

        # Evaluate the network's performance after this training iteration.
        with torch.no_grad():
            # Visualize
            reward_total = {agent: 0.0 for agent in env.agents}
            entropy_total = {agent: 0.0 for agent in env.agents}
            for _ in range(cfg.eval_steps):
                avg_entropy = {agent: 0.0 for agent in env.agents}
                steps_taken = 0
                obs_ = test_env.reset()[0]
                eval_obs = {agent: process_obs(obs_[agent]) for agent in env.agents}
                for _ in range(cfg.max_eval_steps):
                    all_actions = {}
                    for agent in env.agents:
                        distr = Categorical(logits=agents[agent].p_net(eval_obs.unsqueeze(0)).squeeze())
                        action = distr.sample().item()
                        all_actions[agent] = action
                    obs_, reward, eval_done, _, _ = test_env.step(all_actions)
                    eval_obs = {agent: process_obs(obs_[agent]) for agent in env.agents}
                    steps_taken += 1
                    reward_total += reward
                    avg_entropy += distr.entropy()
                    if eval_done:
                        break
                avg_entropy /= steps_taken
                entropy_total += avg_entropy

        wandb.log(
            {
                "avg_eval_episode_reward": reward_total / cfg.eval_steps,
                "avg_eval_entropy": entropy_total / cfg.eval_steps,
            }
        )
