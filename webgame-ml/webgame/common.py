import math
import random
from typing import *
import numpy as np
from webgame_rust import GameState


def process_obs(obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Works for both batched and unbatched inputs.
    """
    scalar_obs, grid_obs = obs[:2]
    grid_shape = grid_obs.shape
    # Add another dim if there's only 2
    add_dim = len(grid_shape) == 2
    if add_dim:
        scalar_obs = scalar_obs[np.newaxis, ...]
        grid_obs = grid_obs[np.newaxis, ...]
        grid_shape = grid_obs.shape
    scalar_obs = np.tile(
        scalar_obs[..., np.newaxis, np.newaxis], [1, 1] + list(grid_shape)[-2:]
    )
    combined = np.concatenate([scalar_obs, grid_obs], 1)
    # If we added another dim, remove it
    if add_dim:
        combined = combined.squeeze(0)
    return combined, obs[2], obs[3]


def pos_to_grid(x: float, y: float, size: int, cell_size: float) -> Tuple[int, int]:
    return (int(round(x / cell_size)), int(round(y / cell_size)))

DIRS = [
    (-1, 0, 7),
    (1, 0, 3),
    (0, 1, 1),
    (0, -1, 5),
]

def explore_policy(game_state: GameState, is_pursuer: bool) -> int:
    from webgame.envs import CELL_SIZE
    if is_pursuer:
        agent_state = game_state.pursuer
        other_state = game_state.player
    else:
        agent_state = game_state.player
        other_state = game_state.pursuer

    # If agents are too close, move away from each other
    dx = agent_state.pos.x - other_state.pos.x
    dy = agent_state.pos.y - other_state.pos.y
    length = math.sqrt(dx**2 + dy**2)
    dx = dx / (length + 0.001)
    dy = dy / (length + 0.001)
    if length <= CELL_SIZE * 2:
        best_dir = max(DIRS, key=lambda x: x[0] * dx + x[1] * dy)
        return best_dir[2]
        
    # Small chance of looking at each other
    if random.random() < 0.1:
        best_dir = max(DIRS, key=lambda x: x[0] * -dx + x[1] * -dy)
        return best_dir[2]
    
    # Otherwise, randomly move away from walls
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
