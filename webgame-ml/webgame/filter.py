from typing import *
import numpy as np
from scipy import signal  # type: ignore

from webgame_rust import AgentState, GameState

from webgame.envs import CELL_SIZE


def pos_to_grid(x: float, y: float, size: int, cell_size: float) -> Tuple[int, int]:
    return (int(round(x / cell_size)), size - int(round(y / cell_size)) - 1)


class BayesFilter:
    """
    A discrete Bayes filter for localization.
    """

    def __init__(self, size: int, cell_size: float):
        self.size = size
        self.cell_size = cell_size
        self.belief = np.ones([size, size]) / size**2

    def localize(self, game_state: GameState, agent_state: AgentState) -> np.ndarray:
        """
        Given an agent's observations, returns the new location probabilities.
        """
        self.belief = self.predict(self.belief)
        self.belief = self.update(self.belief, game_state, agent_state)
        return self.belief

    def predict(self, belief: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = kernel / kernel.sum()
        belief = signal.convolve2d(belief, kernel, mode="same")
        return belief / belief.sum()

    def update(
        self, belief: np.ndarray, game_state: GameState, agent_state: AgentState
    ) -> np.ndarray:
        # Check whether agent can see the player
        player_e, player_obs = list(
            filter(lambda t: t[1].obj_type == "player", game_state.objects.items())
        )[0]
        player_vis_grid = None
        if player_e in agent_state.observing:
            player_vis_grid = pos_to_grid(
                player_obs.pos.x, player_obs.pos.y, self.size, self.cell_size
            )

        obs_grid = np.array(game_state.walls).reshape(
            [game_state.level_size, game_state.level_size]
        )
        posterior = np.zeros([self.size, self.size])
        for y in range(self.size):
            for x in range(self.size):
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
                    agent_lkhd = agent_lkhd * (1.0 / (self.size**2 - sum(agent_state.visible_cells)))
                lkhd = grid_lkhd * agent_lkhd
                posterior[y][x] = lkhd * belief[y][x]
        return posterior / posterior.sum()


if __name__ == "__main__":
    from webgame.envs import ObjsGameEnv
    import rerun as rr  # type: ignore
    import random

    recording_id = "filter_test-" + str(random.randint(0, 10000))
    rr.init(application_id="Pursuer", recording_id=recording_id)
    rr.connect()

    env = ObjsGameEnv(visualize=True, recording_id=recording_id)
    env.reset()

    action_space = env.action_space("pursuer")  # Same for both agents
    assert env.game_state is not None
    b_filter = BayesFilter(env.game_state.level_size, CELL_SIZE)
    for _ in range(100):
        actions = {}
        for agent in env.agents:
            actions[agent] = action_space.sample()
        env.step(actions)[0]

        game_state = env.game_state
        assert game_state is not None
        agent_state = game_state.pursuer
        probs = b_filter.localize(game_state, agent_state)
        rr.log("filter/belief", rr.Tensor(probs), timeless=False)
