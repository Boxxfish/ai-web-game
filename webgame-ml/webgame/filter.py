from typing import *
import numpy as np
from scipy import signal  # type: ignore

import torch
from safetensors.torch import load_model
from webgame_rust import AgentState, GameState

from webgame.common import explore_policy, pos_to_grid, process_obs
from webgame.envs import CELL_SIZE, VisionGameEnv

from torch import nn

from webgame.train_filter import MeasureModel


class BayesFilter:
    """
    A discrete Bayes filter for localization.
    """

    def __init__(
        self,
        size: int,
        cell_size: float,
        update_fn: Callable[
            [np.ndarray, GameState, AgentState, int, float], np.ndarray
        ],
    ):
        self.size = size
        self.cell_size = cell_size
        self.belief = np.ones([size, size]) / size**2
        self.update_fn = update_fn

    def localize(
        self, obs: np.ndarray, game_state: GameState, agent_state: AgentState
    ) -> np.ndarray:
        """
        Given an agent's observations, returns the new location probabilities.
        """
        self.belief = self.predict(self.belief)
        lkhd = self.update_fn(obs, game_state, agent_state, self.size, self.cell_size)
        self.belief = lkhd * self.belief
        self.belief = self.belief / self.belief.sum()
        return self.belief

    def predict(self, belief: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = kernel / kernel.sum()
        belief = signal.convolve2d(belief, kernel, mode="same")
        return belief / belief.sum()


def manual_update(
    obs: np.ndarray,
    game_state: GameState,
    agent_state: AgentState,
    size: int,
    cell_size: float,
) -> np.ndarray:
    # Check whether agent can see the player
    player_e, player_obs = list(
        filter(lambda t: t[1].obj_type == "player", game_state.objects.items())
    )[0]
    player_vis_grid = None
    if player_e in agent_state.observing:
        player_vis_grid = pos_to_grid(
            player_obs.pos.x, player_obs.pos.y, size, cell_size
        )

    obs_grid = np.array(game_state.walls).reshape(
        [game_state.level_size, game_state.level_size]
    )
    lkhd = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            grid_lkhd = 1 - obs_grid[y][x]
            agent_lkhd = 1.0
            if player_vis_grid is not None:
                if player_vis_grid != (x, y):
                    agent_lkhd = 0.0
            else:
                # Cells within vision have 0% chance of agent being there
                agent_lkhd = 1 - int(
                    agent_state.visible_cells[y * game_state.level_size + x]
                )
                # All other cells are equally probable
                agent_lkhd = agent_lkhd * (
                    1.0 / (size**2 - sum(agent_state.visible_cells))
                )
            lkhd[y][x] = grid_lkhd * agent_lkhd
    return lkhd


def model_update(
    model: nn.Module,
) -> Callable[[np.ndarray, GameState, AgentState, int, float], np.ndarray]:
    def model_update_(
        obs: np.ndarray,
        game_state: GameState,
        agent_state: AgentState,
        size: int,
        cell_size: float,
    ) -> np.ndarray:
        with torch.no_grad():
            lkhd = model(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            return lkhd

    return model_update_


if __name__ == "__main__":
    from webgame.envs import ObjsGameEnv
    import rerun as rr  # type: ignore
    import random
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    recording_id = "filter_test-" + str(random.randint(0, 10000))
    rr.init(application_id="Pursuer", recording_id=recording_id)
    rr.connect()

    env = VisionGameEnv(visualize=True, recording_id=recording_id)
    env.reset()

    action_space = env.action_space("pursuer")  # Same for both agents
    assert env.game_state is not None
    if args.checkpoint:
        model = MeasureModel(8, env.game_state.level_size)
        load_model(model, args.checkpoint)
        update_fn = model_update(model)
    else:
        update_fn = manual_update
    b_filter = BayesFilter(env.game_state.level_size, CELL_SIZE, update_fn)
    for _ in range(100):
        actions = {}
        for agent in env.agents:
            if random.random() < 0.1:
                action = action_space.sample()
            else:
                action = explore_policy(env.game_state, agent == "pursuer")
            actions[agent] = action
        obs = process_obs(env.step(actions)[0]["pursuer"])

        game_state = env.game_state
        assert game_state is not None
        agent_state = game_state.pursuer
        lkhd = update_fn(obs, game_state, agent_state, game_state.level_size, CELL_SIZE)
        probs = b_filter.localize(obs, game_state, agent_state)
        rr.log("filter/belief", rr.Tensor(probs), timeless=False)
        rr.log("filter/measurement_likelihood", rr.Tensor(lkhd), timeless=False)
