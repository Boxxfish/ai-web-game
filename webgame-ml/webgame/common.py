from typing import *
import numpy as np

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
    return (int(round(x / cell_size)), size - int(round(y / cell_size)) - 1)