import random
from typing import *
import numpy as np
from webgame_rust import GameState

from webgame.envs import CELL_SIZE


def process_obs(obs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    scalar_obs, grid_obs = obs
    scalar_size = scalar_obs.shape[0]
    grid_shape = grid_obs.shape
    scalar_obs = np.tile(
        scalar_obs[..., np.newaxis, np.newaxis], [1] + list(grid_shape)
    )
    grid_obs = grid_obs[np.newaxis, ...]
    return np.concatenate([scalar_obs, grid_obs], 0)


def pos_to_grid(x: float, y: float, size: int, cell_size: float) -> Tuple[int, int]:
    return (int(round(x / cell_size)), int(round(y / cell_size)))


def explore_policy(game_state: GameState, is_pursuer: bool) -> int:
    if is_pursuer:
        agent_state = game_state.pursuer
    else:
        agent_state = game_state.player

    level_size = game_state.level_size
    tile = pos_to_grid(agent_state.pos.x, agent_state.pos.y, level_size, CELL_SIZE)
    x, y = tile
    walls = game_state.walls
    actions = [
        7,
        3,
        1,
        5
    ]
    mask = [
        x > 0 and not walls[(x - 1) + y * level_size],
        x < level_size - 1 and not walls[(x + 1) + y * level_size],
        y < level_size - 1 and not walls[(x + (y + 1) * level_size)],
        y > 0 and not walls[x + (y - 1) * level_size],
    ]
    valid_actions = []
    for action, mask_val in zip(actions, mask):
        if mask_val:
            valid_actions.append(action)
    if len(valid_actions) == 0:
        return 0
    return random.choice(valid_actions)
