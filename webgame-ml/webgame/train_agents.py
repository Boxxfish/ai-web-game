from argparse import ArgumentParser
from dataclasses import dataclass
from functools import reduce
from typing import Any

import envpool  # type: ignore
import torch
import torch.nn as nn
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from torch.distributions import Categorical
from tqdm import tqdm

from webgame.algorithms.ppo import train_ppo
from webgame.algorithms.rollout_buffer import RolloutBuffer
from webgame.conf import entity
from webgame.utils import init_orthogonal

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
    device: str = "cuda"  # Device to use during training.


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.v_layer1 = nn.Linear(flat_obs_dim, 64)
        self.v_layer2 = nn.Linear(64, 64)
        self.v_layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.v_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.a_layer1 = nn.Linear(flat_obs_dim, 64)
        self.a_layer2 = nn.Linear(64, 64)
        self.a_layer3 = nn.Linear(64, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.a_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


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

    env = envpool.make("CartPole-v1", "gym", num_envs=cfg.num_envs)
    test_env = CartPoleEnv()

    # Initialize policy and value networks
    obs_space = env.observation_space
    act_space = env.action_space
    v_net = ValueNet(obs_space.shape)
    p_net = PolicyNet(obs_space.shape, act_space.n)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=cfg.v_lr)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=cfg.p_lr)

    # A rollout buffer stores experience collected during a sampling run
    buffer = RolloutBuffer(
        obs_space.shape,
        torch.Size((1,)),
        torch.Size((act_space.n,)),
        torch.int,
        cfg.num_envs,
        cfg.train_steps,
    )

    obs = torch.Tensor(env.reset()[0])
    done = False
    for _ in tqdm(range(cfg.iterations), position=0):
        # Collect experience for a number of steps and store it in the buffer
        with torch.no_grad():
            for _ in tqdm(range(cfg.train_steps), position=1):
                action_probs = p_net(obs)
                actions = Categorical(logits=action_probs).sample().numpy()
                obs_, rewards, dones, truncs, _ = env.step(actions)
                buffer.insert_step(
                    obs,
                    torch.from_numpy(actions).unsqueeze(-1),
                    action_probs,
                    rewards,
                    dones,
                    truncs,
                )
                obs = torch.from_numpy(obs_)
                if done:
                    obs = torch.Tensor(env.reset()[0])
                    done = False
            buffer.insert_final_step(obs)

        # Train
        total_p_loss, total_v_loss = train_ppo(
            p_net,
            v_net,
            p_opt,
            v_opt,
            buffer,
            device,
            cfg.train_iters,
            cfg.train_batch_size,
            cfg.discount,
            cfg.lambda_,
            cfg.epsilon,
        )
        buffer.clear()

        # Evaluate the network's performance after this training iteration.
        eval_obs = torch.Tensor(test_env.reset()[0])
        eval_done = False
        with torch.no_grad():
            # Visualize
            reward_total = 0
            entropy_total = 0.0
            eval_obs = torch.Tensor(test_env.reset()[0])
            for _ in range(cfg.eval_steps):
                avg_entropy = 0.0
                steps_taken = 0
                for _ in range(cfg.max_eval_steps):
                    distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
                    action = distr.sample().item()
                    obs_, reward, eval_done, _, _ = test_env.step(action)
                    eval_obs = torch.Tensor(obs_)
                    steps_taken += 1
                    reward_total += reward
                    avg_entropy += distr.entropy()
                    if eval_done:
                        eval_obs = torch.Tensor(test_env.reset()[0])
                        break
                avg_entropy /= steps_taken
                entropy_total += avg_entropy

        wandb.log(
            {
                "avg_eval_episode_reward": reward_total / cfg.eval_steps,
                "avg_eval_entropy": entropy_total / cfg.eval_steps,
                "avg_v_loss": total_v_loss / cfg.train_iters,
                "avg_p_loss": total_p_loss / cfg.train_iters,
            }
        )
