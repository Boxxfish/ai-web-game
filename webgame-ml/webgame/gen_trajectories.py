"""
Script for generating trajectories for training our filter.
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import random
import string
from typing import *
import pickle as pkl

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from webgame.envs import CELL_SIZE, VisionGameEnv
from webgame.filter import pos_to_grid


@dataclass
class TrajDataAll:
    seqs: List[List[np.ndarray]]
    tiles: List[List[int]]


def process_obs(obs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    scalar_obs, grid_obs = obs
    scalar_size = scalar_obs.shape[0]
    grid_shape = grid_obs.shape
    scalar_obs = np.tile(
        scalar_obs[..., np.newaxis, np.newaxis], [1] + list(grid_shape)
    )
    grid_obs = grid_obs[np.newaxis, ...]
    return np.concatenate([scalar_obs, grid_obs], 0)


def main():
    parser = ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="./runs")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-seqs", type=int, default=4_000)
    args = parser.parse_args()

    env = VisionGameEnv()
    action_space = env.action_space("pursuer")
    all_seqs = []
    all_tiles = []
    for seq_idx in tqdm(range(args.num_seqs)):
        obs, infos = env.reset()
        seq = []
        tiles = []
        for seq_step in range(args.seq_len):
            actions = {}
            for agent in env.agents:
                actions[agent] = action_space.sample()
            obs, rewards_, dones_, truncs_, infos = env.step(actions)
            pursuer_obs = obs["pursuer"]
            processed_obs = process_obs(pursuer_obs)
            player_pos = env.game_state.player.pos
            gold_tile = pos_to_grid(
                player_pos.x, player_pos.y, env.game_state.level_size, CELL_SIZE
            )
            seq.append(process_obs)
            tiles.append(gold_tile)
        all_seqs.append(seq)
        all_tiles.append(tiles)

    out_dir = Path(args.out_dir)
    out_id = "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
    )
    traj_data_all = TrajDataAll(seqs=all_seqs, tiles=all_tiles)
    os.mkdir(out_dir / out_id)
    with open(out_dir / out_id / "traj_data_all.pkl", "wb") as f:
        pkl.dump(traj_data_all, f)


if __name__ == "__main__":
    main()